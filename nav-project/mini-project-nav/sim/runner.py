"""The closed loop: render what the robot sees, ask the policy for a command, advance the physics, repeat."""
import importlib.util
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from .episode import Episode, PHYS_HZ, POLICY_HZ, to_robot_frame, wrap
from .splat import Splats, pick_device

HISTORY = 25            # camera frames kept for the policy (6 s at 4 Hz)


@dataclass
class Observation:
    """Everything a policy may use. Note that the route is not in here: only its end is, as a point and as a picture."""
    image: np.ndarray       # camera image now, uint8 [H, W, 3], RGB (224 x 126, 90 degrees horizontal field of view)
    frames: list            # camera images at 4 Hz (0.25 s apart), oldest first, up to HISTORY of them; the last one is `image`
    goal: tuple             # end of the route as (forward, left) in meters, in the robot frame
    goal_image: np.ndarray  # what the camera sees standing at the end of the route, looking along it: uint8 [H, W, 3], the same for the whole episode
    goal_distance: float    # straight-line distance to that point, meters
    odometry: np.ndarray    # past robot poses (forward, left, yaw) in the current robot frame, 4 Hz; the last row is (0, 0, 0)
    speed: float            # m/s
    yaw_rate: float         # rad/s
    t: float                # seconds since the episode started


class Renderer:
    """Renders the robot's camera view. Keeps the splats of the last `keep` scenes in memory, since loading one is slow."""

    def __init__(self, camera, device="auto", keep=2):
        self.camera, self.device, self.keep = camera, pick_device(device), keep
        self._cache = {}                                    # scene id -> Splats, oldest first

    def image(self, scene, x, y, yaw):
        """uint8 [H, W, 3] view from the robot standing at (x, y) with heading yaw."""
        if scene.id not in self._cache:
            if len(self._cache) >= self.keep:
                self._cache.pop(next(iter(self._cache)))
            self._cache[scene.id] = Splats(os.path.join(scene.dir, "splat.ply"), self.device)
        cam = self.camera
        pose = scene.camera_pose(x, y, yaw, cam["height_m"])
        return self._cache[scene.id].render(pose, cam["width"], cam["height"], cam["fov"])


def load_policy(path):
    """Import the Python file at `path` and return an instance of the `Policy` class it defines."""
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("student_policy", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))               # so the policy can import the files next to it (your model code)
    spec.loader.exec_module(module)
    if not hasattr(module, "Policy"):
        raise RuntimeError(f"{path} has no class named Policy")
    return module.Policy()


def observe(renderer, scene, episode, frames, poses, goal_image):
    """Render the current view and build the policy's Observation.

    `frames` and `poses` hold the history so far and are extended in place.
    """
    s = episode.state
    frames.append(renderer.image(scene, s.x, s.y, s.yaw))
    poses.append((s.x, s.y, s.yaw))
    del frames[:-HISTORY], poses[:-HISTORY]

    odometry = []
    for x, y, yaw in poses:
        forward, left = to_robot_frame(x, y, s)
        odometry.append((forward, left, wrap(yaw - s.yaw)))
    goal_forward, goal_left = episode.goal_in_robot_frame()
    return Observation(image=frames[-1], frames=list(frames), goal=(float(goal_forward), float(goal_left)), goal_image=goal_image,
                       goal_distance=episode.goal_distance(), odometry=np.array(odometry, np.float32),
                       speed=float(s.v), yaw_rate=float(s.w), t=float(episode.t))


def run_episode(renderer, scene, spec, policy, on_step=None, should_stop=None, realtime=0.0):
    """Run one episode and return (result dict, Episode).

    on_step(episode, image) is called after every physics step; `image` is the policy's camera image on the steps where
    the policy was just called, otherwise None. should_stop() is polled every step and ends the episode early.
    realtime=1 slows the loop to real time; 0 runs as fast as possible.
    """
    episode = Episode(scene, spec)
    goal_image = renderer.image(scene, *episode.goal_pose())            # the picture of the end of the route, rendered once
    episode.goal_image = goal_image                         # kept for the video
    frames, poses = [], []
    policy.reset(dict(scene=scene.id, camera=renderer.camera, hz=POLICY_HZ, time_limit=episode.time_limit))

    command = (0.0, 0.0)                                    # (speed, yaw rate), held between policy calls
    tick = 0
    deadline = time.time()                                  # when the next physics step is due (real-time mode)
    while not episode.done:
        if should_stop is not None and should_stop():
            episode.done, episode.reason = True, "stopped"
            break
        image = None
        if tick % (PHYS_HZ // POLICY_HZ) == 0:
            obs = observe(renderer, scene, episode, frames, poses, goal_image)
            image = obs.image
            out = policy.act(obs)
            try:
                command = (float(out[0]), float(out[1]))
            except Exception as e:
                raise RuntimeError(f"Policy.act must return (v, w), got {out!r}") from e
        episode.command = command
        episode.step(*command)
        tick += 1
        if on_step is not None:
            on_step(episode, image)
        if realtime > 0:                                    # never faster than the clock; at most one policy period of lost time is made up
            deadline += 1.0 / (PHYS_HZ * realtime)
            now = time.time()
            if deadline > now:
                time.sleep(deadline - now)
            else:
                deadline = max(deadline, now - 1.0 / (POLICY_HZ * realtime))
    return episode.result(), episode
