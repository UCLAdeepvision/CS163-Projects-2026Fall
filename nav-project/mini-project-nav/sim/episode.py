"""The robot, its physics, the rules of one episode, and the score.

The robot is a car-like base, like the COCO delivery robot in coco-gs: it steers with its front axle and cannot turn on the spot.
A policy commands a forward speed v (m/s) and a yaw rate w (rad/s, positive turns left). The yaw rate is converted to the steering
angle that produces it at the current speed, w = v * tan(steering angle) / WHEEL_BASE, and the steering angle is limited in
range and in rate. So the tightest turn has a radius of WHEEL_BASE / tan(MAX_STEER), about 1.25 m, and a robot that is not
moving cannot change its heading.

Physics runs at 40 Hz. The policy is called at 4 Hz, so each command is held for ten physics steps.
When the robot's box touches the scene mesh the motion is cancelled and the robot stops, and the episode ends as a failure.

World frame: meters, x and y on the ground plane, yaw measured counter-clockwise from +x.
Robot frame: x forward, y left.
"""
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

# Timing
PHYS_HZ = 40
POLICY_HZ = 4

# Robot limits and response. WHEEL_BASE, MAX_STEER, MAX_SPEED and STEER_RATE are the values of the COCO robot in coco-gs
# (urbansim/primitives/robot/coco_new.py); MAX_REVERSE and TAU_SPEED are ours.
WHEEL_BASE = 0.456          # m between the front and the rear axle
MAX_STEER = 0.35            # rad, largest steering angle of the front axle (20 degrees)
STEER_RATE = 1.75           # rad/s, fastest change of the steering angle (measured on the real robot)
MAX_SPEED = 2.5             # m/s forward
MAX_REVERSE = 0.5           # m/s backward
TAU_SPEED = 0.25            # s, time constant of the speed lag
MIN_STEERING_SPEED = 0.1    # m/s: below this speed a yaw rate command is ignored and the steering angle is left as it is

# Episode rules
SUCCESS_RADIUS = 0.5        # m of route left to cover; at or below this the episode is a success
FREEZE_WINDOW = 5.0         # s
FREEZE_PROGRESS = 0.1       # m of route progress required per FREEZE_WINDOW, otherwise the episode ends
MIN_TIME_LIMIT = 45.0       # s
TIME_PER_METER = 2.5        # s of allowed time per meter of route (never less than MIN_TIME_LIMIT)


def wrap(angle):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0          # forward speed
    steer: float = 0.0      # steering angle of the front axle (rad); keeps its value through a collision
    w: float = 0.0          # yaw rate of the last step that actually moved the robot (0 after a collision)


def to_robot_frame(px, py, state):
    """Express the world point (px, py) as (forward, left) in the frame of `state`."""
    dx, dy = px - state.x, py - state.y
    c, s = np.cos(state.yaw), np.sin(state.yaw)
    return c * dx + s * dy, -s * dx + c * dy


class Episode:
    """One run of the robot along one route.

    `spec` is an entry of the benchmark file: start position, start yaw, and the route as a polyline ("centerline").
    The route is only used for scoring and for the goal (its last point); a policy never sees it.
    Call step() repeatedly until `done` is set; result() then describes the outcome.
    """

    def __init__(self, scene, spec):
        self.scene = scene
        self.spec = spec
        self.state = RobotState(spec["start"][0], spec["start"][1], spec["yaw"])

        # route geometry: the polyline and the arc length at each of its points
        self.line = np.array(spec["centerline"], np.float64)
        self.cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.line, axis=0), axis=1))]
        self.route_len = float(self.cum[-1])
        self.goal = self.line[-1]
        self.time_limit = max(MIN_TIME_LIMIT, TIME_PER_METER * self.route_len)

        # progress along the route: arc length of the farthest route point reached so far
        self.progress = 0.0
        self._nearest = 0.0                     # arc length of the route point nearest the robot
        self._times = [0.0]                     # history of (time, progress), used by the "no progress" rule
        self._progress = [0.0]

        # bookkeeping
        self.t = 0.0
        self.command = (0.0, 0.0)               # the (speed, yaw rate) the policy last asked for; kept for the video
        self.collisions = 0                     # 0 or 1: the first contact with the scene ends the episode
        self.colliding = False                  # in contact during the latest step
        self.done = False
        self.reason = ""                        # why the episode ended
        self.path_len = 0.0                     # distance the robot actually drove
        self.trace = [(self.state.x, self.state.y, self.state.yaw)]
        self.collision_points = []

    # ----- what the policy is told ------------------------------------------------------------------------------------

    def goal_point(self):
        """The point the policy is told to drive to: the end of the route."""
        return self.goal

    def goal_pose(self):
        """(x, y, yaw) of the camera that takes the goal image: at the end of the route, looking along its last meter."""
        tail = self.line[self.cum >= self.route_len - 1.0]
        d = tail[-1] - tail[0] if len(tail) >= 2 else self.line[-1] - self.line[-2]
        return float(self.goal[0]), float(self.goal[1]), float(np.arctan2(d[1], d[0]))

    def goal_in_robot_frame(self):
        goal = self.goal_point()
        return to_robot_frame(goal[0], goal[1], self.state)

    def goal_distance(self):
        goal = self.goal_point()
        return float(np.hypot(goal[0] - self.state.x, goal[1] - self.state.y))

    # ----- one physics step ---------------------------------------------------------------------------------------------

    def step(self, v_cmd, w_cmd, dt=1.0 / PHYS_HZ):
        """Advance the simulation by dt seconds while holding the command (v_cmd, w_cmd)."""
        if self.done:
            return
        x, y, yaw, w = self._integrate(v_cmd, w_cmd, dt)
        self.t += dt
        self._resolve_contact(x, y, yaw, w)

        s = self.state
        self.trace.append((s.x, s.y, s.yaw))
        self.progress = max(self.progress, self._project())
        self._times.append(self.t)
        self._progress.append(self.progress)
        self._check_end()

    def _integrate(self, v_cmd, w_cmd, dt):
        """Apply the speed lag and the steering limits, and return the pose the robot would reach if nothing were in the way.

        The pose is the center of the robot. It moves along a bicycle model: the rear axle drives straight, the front axle
        is turned by the steering angle, and the center slides sideways by the slip angle beta.
        """
        s = self.state
        v_cmd = float(np.clip(v_cmd if np.isfinite(v_cmd) else 0.0, -MAX_REVERSE, MAX_SPEED))
        w_cmd = float(w_cmd if np.isfinite(w_cmd) else 0.0)
        s.v += (v_cmd - s.v) * min(1.0, dt / TAU_SPEED)

        # steering angle that gives the commanded yaw rate at the commanded speed; a robot that is not moving keeps its angle
        if abs(v_cmd) >= MIN_STEERING_SPEED:
            target = float(np.clip(np.arctan(w_cmd * WHEEL_BASE / v_cmd), -MAX_STEER, MAX_STEER))
        else:
            target = s.steer
        s.steer = float(np.clip(s.steer + np.clip(target - s.steer, -STEER_RATE * dt, STEER_RATE * dt), -MAX_STEER, MAX_STEER))

        beta = np.arctan(0.5 * np.tan(s.steer))
        w = s.v * np.cos(beta) * np.tan(s.steer) / WHEEL_BASE
        heading = s.yaw + 0.5 * w * dt + beta
        return s.x + s.v * np.cos(heading) * dt, s.y + s.v * np.sin(heading) * dt, wrap(s.yaw + w * dt), w

    def _resolve_contact(self, x, y, yaw, w):
        """Move the robot to the new pose, unless its box touches the mesh there: then it stays and stops."""
        s = self.state
        if self.scene.in_collision(x, y, yaw):
            if not self.colliding:
                self.collisions += 1
                self.collision_points.append((s.x, s.y))
            self.colliding = True
            s.v, s.w = 0.0, 0.0
        else:
            self.colliding = False
            self.path_len += float(np.hypot(x - s.x, y - s.y))
            s.x, s.y, s.yaw, s.w = x, y, yaw, w

    def _project(self):
        """Arc length of the route point nearest the robot.

        Only a short window around the previous answer is searched, so that a route which doubles back on itself cannot be
        shortcut by jumping to a later part of it.
        """
        window = np.nonzero((self.cum >= self._nearest - 1.0) & (self.cum <= self._nearest + 6.0))[0]
        d = np.hypot(self.line[window, 0] - self.state.x, self.line[window, 1] - self.state.y)
        self._nearest = float(self.cum[window[int(d.argmin())]])
        return self._nearest

    def _progress_at(self, t):
        return self._progress[min(bisect_left(self._times, t), len(self._times) - 1)]

    def _check_end(self):
        s = self.state
        if self.route_len - self.progress <= SUCCESS_RADIUS:
            self.done, self.reason = True, "goal reached"
        elif self.collisions > 0:
            self.done, self.reason = True, "collision"
        elif not self.scene.on_walkable(s.x, s.y):
            self.done, self.reason = True, "left the sidewalk"
        elif self.t >= self.time_limit:
            self.done, self.reason = True, "time limit"
        elif self.t > FREEZE_WINDOW and self.progress - self._progress_at(self.t - FREEZE_WINDOW) < FREEZE_PROGRESS:
            self.done, self.reason = True, f"no progress for {FREEZE_WINDOW:.0f} s"

    # ----- outcome ------------------------------------------------------------------------------------------------------

    def result(self):
        success = self.reason == "goal reached"
        completion = 1.0 if success else float(np.clip(self.progress / self.route_len, 0.0, 1.0))
        return dict(scene=self.scene.id, route=self.spec["route_id"], success=success, route_completion=round(completion, 3),
                    collisions=int(self.collisions), time=round(self.t, 2), path_length=round(self.path_len, 2), reason=self.reason)


def episode_score(result):
    """Score of one episode: 1 if the robot reached the goal without touching anything, otherwise 0."""
    return 1.0 if result["success"] else 0.0


def summarize(results):
    """Benchmark summary of a list of episode results.

    score             percentage of the episodes that reached the goal without a collision (this is the number that counts)
    route_completion  mean fraction of the route covered, for information
    collision_rate    fraction of the episodes that ended in a collision, for information
    """
    n = max(len(results), 1)
    return dict(score=round(100 * sum(episode_score(r) for r in results) / n, 2),
                route_completion=round(sum(r["route_completion"] for r in results) / n, 3),
                collision_rate=round(sum(r["collisions"] > 0 for r in results) / n, 3), episodes=len(results))
