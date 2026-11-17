"""The policy that scores a submitted ONNX file. This is what the course staff run on the held-out episodes.

    python evaluate.py --policy code/onnx_policy.py --onnx submission.onnx

The file is run with ONNX Runtime on the CPU, so it can contain any network that meets the interface described in export_onnx.py.
The simulator's camera images and the picture of the goal go in as they are; the waypoints that come out are followed by the PD
controller in policy.py. Changing it there changes your own runs but not the grading, which uses the unmodified code.
"""
import os

import numpy as np
import onnxruntime

from policy import PDController, stack_images


class Policy:
    def __init__(self):
        path = os.environ.get("SUBMISSION_ONNX")
        if not path:
            raise RuntimeError("no model given: run evaluate.py with --onnx submission.onnx")
        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.frames = {i.name: i for i in self.session.get_inputs()}["images"].shape[1]       # how many camera images this model reads
        self.controller = PDController()

    def reset(self, info):
        """Called at the start of every episode."""
        self.controller = PDController()

    def act(self, obs):
        """Return (speed in m/s, yaw rate in rad/s) for the observation `obs`."""
        feed = {"images": stack_images(obs, self.frames), "goal_image": obs.goal_image[None], "goal": np.array([obs.goal], np.float32)}
        waypoints = self.session.run(["waypoints"], feed)[0][0]     # [5, 2], meters
        self.waypoints = waypoints                                  # kept so that the video can draw them
        return self.controller(waypoints)
