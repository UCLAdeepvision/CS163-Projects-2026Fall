"""Part 4: real robot data (cameras + LiDAR).  Train on drive 1 (pass 1); evaluate on
  val_pass2       - the front camera on a SECOND DRIVE over the same route (registered into drive 1's frame; THE TEST),

    python part4/train.py --out runs/p4_base                          # the baseline: random initial Gaussians
    python part4/train.py --init file --init-file my_init.npz --out runs/p4_mine   # your own initialisation (npz with xyz [N,3], rgb [N,3] uint8)

The baseline starts from N points drawn uniformly in a cube around the training cameras (random colours); the cube's half-width is
`--cube` x the scene scale.  Everything else (what the initial Gaussians are, extra losses, ...) is up to you.
"""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gsmini import paths as _paths, friendly as _friendly; _friendly.install()
from gsmini import Config, Scene, Trainer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=_paths.data_dir("part4_robot"))
ap.add_argument("--init", default="random", choices=["random", "file"])
ap.add_argument("--init-file", default="", help="with --init file: an .npz with xyz [N,3] float and rgb [N,3] uint8, in the scene's world frame")
ap.add_argument("--n-init", type=int, default=300000, help="number of random initial points")
ap.add_argument("--cube", type=float, default=1.0, help="random init: half-width of the cube, in units of the scene scale")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

scene = Scene(a.data)
TRAIN = "train_front"                                          # the training views: the front camera of drive 1
if a.init == "random":
    ids = scene.indices(TRAIN)
    c = np.stack([scene.frames[i].c2w[:3, 3] for i in ids]); centre = c.mean(0); scale = scene.scene_scale(TRAIN)
    rng = np.random.default_rng(a.seed)
    scene.init_points = {"xyz": (centre + a.cube * scale * (rng.random((a.n_init, 3)) * 2 - 1)).astype(np.float32),
                         "rgb": (rng.random((a.n_init, 3)) * 255).astype(np.uint8)}
    print(f"random init: {a.n_init:,} points in a cube of half-width {a.cube * scale:.1f} m around the training cameras")
else:
    p = np.load(a.init_file); scene.init_points = {"xyz": p["xyz"].astype(np.float32), "rgb": p["rgb"].astype(np.uint8)}
    print(f"init from {a.init_file}: {len(p['xyz']):,} points")
cfg = Config(steps=a.steps, train_split=TRAIN, eval_splits=["val_pass2"], seed=a.seed)


Trainer(scene, cfg, a.out).train()
