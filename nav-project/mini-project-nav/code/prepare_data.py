"""Turn the raw clips into one array of small frames and one array of poses.

    python code/prepare_data.py --clips /path/to/clips --out data/processed

Each clip is a 20 second fisheye video (front_fe_384x216.mp4, 20 Hz) plus the robot's pose for every frame. For each clip this
keeps every 5th frame (4 Hz, the rate the simulator runs at), undistorts it to the 90 degree pinhole camera that the simulator renders, and
maps it onto the camera ViNT was trained on (see vint.align) at the 85 x 64 pixels that ViNT reads. Writes to the output folder
    frames.npy   uint8 [total frames, 64, 85, 3]
    poses.npy    float64 [total frames, 3]: x, y (meters), yaw (radians)
    index.json   one entry per clip: its name, its recording, the row where it starts in the arrays, and its length
"""
import argparse
import glob
import json
import os
import subprocess
from multiprocessing import Pool

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

from vint import HEIGHT, WIDTH, affine_for

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
STRIDE = 5          # keep every 5th frame: 20 Hz -> 4 Hz
VIDEO_WIDTH, VIDEO_HEIGHT = 384, 216

# Fisheye calibration of the 1920x1080 camera (OpenCV fisheye model), with K scaled to the 384x216 video (factor 0.2).
K = np.array([[158.0, 0, 192.0], [0, 158.0, 108.0], [0, 0, 1]])
D = np.array([-0.03054249, 0.00974967, -0.00679687, 0.00238449])
# The pinhole camera to render into: the same image size, 90 degrees horizontal field of view.
F_PINHOLE = (VIDEO_WIDTH / 2) / np.tan(np.radians(45))
P = np.array([[F_PINHOLE, 0, VIDEO_WIDTH / 2], [0, F_PINHOLE, VIDEO_HEIGHT / 2], [0, 0, 1]])
# For every pixel of the pinhole image, where to look in the fisheye image.
MAP1, MAP2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (VIDEO_WIDTH, VIDEO_HEIGHT), cv2.CV_32FC1)
# From the undistorted 384 x 216 picture (focal length F_PINHOLE = 192 pixels) to ViNT's 85 x 64: the same map `align` applies to a rendered image.
a, c, e, f = affine_for(VIDEO_WIDTH, VIDEO_HEIGHT, F_PINHOLE)
AFFINE = (a, 0.0, c, 0.0, e, f)


def read_clip(folder):
    """Return (frames [n, 64, 85, 3], poses [n, 3]) of one clip folder, at 4 Hz."""
    position = np.load(f"{folder}/frame_positions.npy")[:, :2]
    quat = np.load(f"{folder}/frame_orientations.npy")                      # w, x, y, z
    # Heading from the quaternion (rotation about the vertical axis). The file frame_yaw_deg.npy is not the direction of travel.
    yaw = np.unwrap(2 * np.arctan2(quat[:, 3], quat[:, 0]))
    poses = np.stack([position[:, 0], position[:, 1], yaw], 1)[::STRIDE]

    # decode the video with ffmpeg, keeping every STRIDE-th frame
    video = subprocess.run([FFMPEG, "-v", "error", "-i", f"{folder}/front_fe_384x216.mp4", "-vf", f"select=not(mod(n\\,{STRIDE}))",
                            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(video, np.uint8).reshape(-1, VIDEO_HEIGHT, VIDEO_WIDTH, 3)
    frames = np.stack([np.array(Image.fromarray(cv2.remap(f, MAP1, MAP2, cv2.INTER_LINEAR)).transform(
        (WIDTH, HEIGHT), Image.AFFINE, AFFINE, resample=Image.BILINEAR, fillcolor=0)) for f in frames])
    n = min(len(frames), len(poses))                                        # the video and the pose file can differ by a frame
    return frames[:n], poses[:n]


def read_clip_safe(folder):
    """read_clip for a worker process: returns (name, frames, poses), or (name, None, error message) if the clip is broken."""
    try:
        return os.path.basename(folder), *read_clip(folder)
    except Exception as e:
        return os.path.basename(folder), None, str(e)


def recording_of(name):
    """The recording a clip belongs to. Clips of one recording are never split between training and validation.

    Names are either date__device__recording__..., or start with the device id (C####...).
    """
    parts = name.split("__")
    return parts[2] if name.startswith("20") else parts[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="folder that contains one subfolder per clip")
    ap.add_argument("--out", required=True, help="folder to write frames.npy, poses.npy and index.json to")
    ap.add_argument("--workers", type=int, default=4, help="number of processes decoding clips in parallel")
    args = ap.parse_args()

    folders = sorted(d for d in glob.glob(args.clips + "/*") if os.path.exists(d + "/front_fe_384x216.mp4"))
    frames, poses, index, failed = [], [], [], []
    start = 0                                                               # row of the next clip in the big arrays
    with Pool(args.workers) as pool:
        for name, f, p in pool.imap(read_clip_safe, folders, chunksize=4):
            if f is None:
                failed.append((name, p))
                continue
            frames.append(f)
            poses.append(p)
            index.append(dict(clip=name, start=start, n=len(f), recording=recording_of(name)))
            start += len(f)
            if len(index) % 200 == 0:
                print(len(index), "clips", flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/frames.npy", np.concatenate(frames))
    np.save(f"{args.out}/poses.npy", np.concatenate(poses))
    with open(f"{args.out}/index.json", "w") as f:
        json.dump(index, f)
    print(f"{len(index)} clips, {start} frames, {len(failed)} failed")
    for name, error in failed:
        print("  failed:", name, error)


if __name__ == "__main__":
    main()
