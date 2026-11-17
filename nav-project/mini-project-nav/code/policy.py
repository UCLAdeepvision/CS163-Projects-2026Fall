"""The policy the simulator runs: ViNT predicts waypoints from the camera images and a picture of the goal, and the shared PD controller of
SidewalkBench-GS turns them into a speed and a yaw rate.

The controller is the same for every model; it only needs the waypoints and their spacing in time. The simulator steers the
robot like a car and converts the yaw rate into a steering angle. It calls Policy.act(obs) four times per second and expects a
(speed, yaw rate) pair back. `obs` is a sim.runner.Observation; this policy uses its camera frames and its goal image. Fine-tuned weights are
passed through the CHECKPOINT environment variable, which `evaluate.py --checkpoint` sets.
"""
import math
import os

import numpy as np
import torch

from vint import SPACING, WAYPOINTS, Waypoints, load_model, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))

FRAMES = 6              # camera images the network reads (0.25 s apart)


def stack_images(obs, count):
    """The last `count` camera images of an Observation as one uint8 array [1, count, H, W, 3], oldest first.
    At the start of an episode fewer images exist, so the first one is repeated."""
    images = list(obs.frames[-count:])
    images = [images[0]] * (count - len(images)) + images
    return np.stack(images)[None]


class Policy:
    def __init__(self):
        self.device = pick_device("mps" if torch.backends.mps.is_available() else "auto")
        model = load_model(os.environ.get("CHECKPOINT"), self.device)
        self.net = Waypoints(model).to(self.device).eval()
        self.controller = PDController()

    def reset(self, info):
        """Called at the start of every episode. `info` describes the scene."""
        self.controller = PDController()

    @torch.no_grad()
    def act(self, obs):
        """Return (speed in m/s, yaw rate in rad/s) for the observation `obs`."""
        images = torch.from_numpy(stack_images(obs, FRAMES)).to(self.device)
        goal_image = torch.from_numpy(obs.goal_image[None]).to(self.device)
        goal = torch.tensor([obs.goal], dtype=torch.float32, device=self.device)
        waypoints = self.net(images, goal_image, goal)[0].cpu().numpy()          # [5, 2], meters, robot frame
        self.waypoints = waypoints                                                # kept so that the video can draw them
        return self.controller(waypoints)


class PDController:
    """The shared PD controller of SidewalkBench-GS (sidewalkbench/models/pd_controller.py) in its default "half horizon" mode.

    The waypoints are where the robot should be SPACING, 2 * SPACING, ... seconds from now, so the prediction horizon is
    WAYPOINTS * SPACING = 1.25 s. The speed comes from the waypoint nearest half the horizon: its forward distance divided by that half
    horizon, which is the average speed the model asks for. The yaw rate comes from the bearing of the last waypoint divided by the same
    time. The robot never reverses, the speed may rise by at most ACCEL and fall by at most BRAKE per call, and both commands are
    clamped. The simulator turns (speed, yaw rate) into a steering angle, so the robot steers like the real one and cannot turn
    without moving.
    """
    MAX_V = 1.5             # m/s
    MAX_W = 0.65            # rad/s
    ACCEL = 0.4             # m/s per call
    BRAKE = 0.5             # m/s per call

    def __init__(self, count=WAYPOINTS, spacing=SPACING):
        times = spacing * np.arange(1, count + 1)
        self.dt = float(times[-1] / 2)                                    # half the horizon, in seconds
        self.speed_index = int(np.argmin(np.abs(times - self.dt)))        # the waypoint nearest that time (the earlier one if two are equally near)
        self.last_v = None

    def __call__(self, waypoints):
        """waypoints: [K, 2] predicted positions (forward, left) in meters. Returns (speed, yaw rate)."""
        x = float(waypoints[self.speed_index][0])
        x_last, y_last = float(waypoints[-1][0]), float(waypoints[-1][1])

        v = max(x / self.dt, 0.0)
        w = math.atan2(y_last, max(abs(x_last), 1e-6)) * (1.0 if x_last >= 0 else -1.0) / self.dt
        if abs(x) < 1e-6:
            v, w = 0.0, math.copysign(math.pi / 20.0, y_last)
        if self.last_v is not None:
            v = max(min(max(v, self.last_v - self.BRAKE), self.last_v + self.ACCEL), 0.0)
        v = min(max(v, 0.0), self.MAX_V)
        w = min(max(w, -self.MAX_W), self.MAX_W)
        self.last_v = v
        return v, w
