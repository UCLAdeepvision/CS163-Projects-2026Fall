"""One training sample, and the train / validation / test split.

A sample is what the model sees and should predict at one moment of one clip:
    obs      the last six camera frames (0.25 s apart)
    goal     a later camera frame of the same clip: the picture of the place the robot is heading for
    target   where the robot really was 0.25, 0.5, ... 1.25 s later, in the robot's frame

Distances are in model units, STEP meters each (see vint.py).
"""
import hashlib
import json

import cv2
import numpy as np
import torch

from vint import CONTEXT, MEAN, STD, STEP, WAYPOINTS

MIN_MOVE = 0.15         # meters the robot must move in 1.25 s for a moment to be used (not standing still)


def split_of(recording, val=0.1, test=0.1):
    """Which split a recording belongs to: "train", "val" or "test".

    The choice depends only on the recording's name, so all of its clips land in the same split. Splitting by clip or by frame
    would put near-identical frames into both training and validation.
    """
    h = int(hashlib.md5(recording.encode()).hexdigest(), 16) % 1000 / 1000
    return "test" if h < test else "val" if h < test + val else "train"


def world_to_robot(points, pose):
    """Express world positions [..., 2] in the frame of `pose` = (x, y, yaw): x forward, y left."""
    d = points[..., :2] - pose[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def normalize(frame):
    """uint8 [64, 85, 3] -> float32 [3, 64, 85], with the ImageNet statistics ViNT expects."""
    return ((frame.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


class ClipDataset(torch.utils.data.Dataset):
    """Yields (obs [18, 64, 85], goal [3, 64, 85], target [5, 2]) for the clips of one split.

    root         the folder written by prepare_data.py
    split        "train", "val" or "test"
    goal_steps   (min, max): the goal picture is the frame this many steps (0.25 s each) after the current one, capped at the end of the
                 clip. Training draws a random number from this range for every sample; validation always uses the middle of it.
    augment      True for training: random samples, left-right mirroring, brightness changes.
                 False for validation: every valid moment once, in order.
    """

    def __init__(self, root, split, goal_steps=(3, 14), augment=True, epoch_len=60000):
        self.root = root
        self._frames = None                                 # opened on first use, so workers started with "spawn" (macOS) don't copy it
        self.poses = np.load(f"{root}/poses.npy")           # [total frames, 3]: x, y, yaw
        self.clips = [c for c in json.load(open(f"{root}/index.json")) if split_of(c["recording"]) == split]
        self.goal_steps, self.augment, self.epoch_len = goal_steps, augment, epoch_len

        # the usable moments: (clip, frame) pairs from which the robot moves, with room for the goal picture
        self.items = []
        for ci, clip in enumerate(self.clips):
            p = self.poses[clip["start"]:clip["start"] + clip["n"], :2]
            for t in range(CONTEXT - 1, clip["n"] - max(goal_steps[0], WAYPOINTS) - 1):
                if np.linalg.norm(p[t + WAYPOINTS] - p[t]) >= MIN_MOVE:
                    self.items.append((ci, t))

    @property
    def frames(self):
        """All frames of the dataset [total frames, 64, 85, 3], memory-mapped from frames.npy."""
        if self._frames is None:
            self._frames = np.load(f"{self.root}/frames.npy", mmap_mode="r")
        return self._frames

    def __len__(self):
        return self.epoch_len if self.augment else len(self.items)

    def __getitem__(self, i):
        # training draws random samples (i is ignored); validation walks the list in order
        rng = np.random.default_rng() if self.augment else np.random.default_rng(i)
        ci, t = self.items[rng.integers(len(self.items))] if self.augment else self.items[i]
        clip = self.clips[ci]
        poses = self.poses[clip["start"]:clip["start"] + clip["n"]]
        low, high = self.goal_steps
        high = min(high, clip["n"] - 1 - t)                                         # the goal frame must exist
        steps = int(rng.integers(low, high + 1)) if self.augment else min((low + self.goal_steps[1]) // 2, high)
        first = clip["start"] + t - CONTEXT + 1
        obs = np.array(self.frames[first:first + CONTEXT])                          # [6, 64, 85, 3], oldest first
        goal = np.array(self.frames[clip["start"] + t + steps])
        target = (world_to_robot(poses[t + np.arange(1, WAYPOINTS + 1)], poses[t]) / STEP).astype(np.float32)

        if self.augment:
            if rng.random() < 0.5:                          # mirror left and right: flip the pictures and y
                obs, goal, target = obs[:, :, ::-1], goal[:, ::-1], target * np.array([1, -1], np.float32)
            gain, bias = rng.uniform(0.75, 1.25), rng.uniform(-20, 20)             # contrast and brightness, the same for all pictures
            obs, goal = np.clip(obs * gain + bias, 0, 255), np.clip(goal * gain + bias, 0, 255)
        obs = np.concatenate([normalize(np.ascontiguousarray(f).astype(np.uint8)) for f in obs], 0)
        goal = normalize(np.ascontiguousarray(goal).astype(np.uint8))
        return torch.from_numpy(obs), torch.from_numpy(goal), torch.from_numpy(np.ascontiguousarray(target))
