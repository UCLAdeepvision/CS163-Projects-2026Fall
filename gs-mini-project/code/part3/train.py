"""Part 3: a better training recipe for a street scene (Wanderland, handheld capture).  Held-out evaluation on val_offtraj (frames from a
different trajectory than the training walk).  All metrics are computed over the unmasked pixels.

    python part3/train.py --out runs/p3_base                      # the baseline: plain 3DGS with the default settings below
    python part3/train.py --refine-every 500 --out runs/p3_try1   # change a setting, compare with the baseline

The defaults of every flag ARE the baseline.  Your job is to find settings that train a better model:
  densification (how Gaussians are added and removed during training):
    --refine-every N    check for Gaussians to split / duplicate / remove every N steps
    --reset-every N     set all opacities back to a low value every N steps (a huge number = never)
  training length:
    --steps N           number of training steps (default 7000; the baseline of the goal check must use the default)
  camera poses:
    --pose-opt          learn small corrections of the camera poses together with the model (--pose-lr sets how fast)
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsmini import Config, Scene, Trainer, paths as _paths, friendly as _friendly
_friendly.install()

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default=_paths.data_dir("part3_wanderland"))
ap.add_argument("--steps", type=int, default=7000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sh-degree", type=int, default=3, help="degree of the spherical-harmonics colour (0 = view-independent)")
ap.add_argument("--refine-every", type=int, default=100)
ap.add_argument("--reset-every", type=int, default=3000)
ap.add_argument("--pose-opt", action="store_true", help="learn per-frame + per-camera pose corrections")
ap.add_argument("--pose-lr", type=float, default=2e-5)
ap.add_argument("--out", required=True)
a = ap.parse_args()

scene = Scene(a.data)
cfg = Config(steps=a.steps, eval_splits=["val_offtraj"], seed=a.seed, sh_degree=a.sh_degree,
             refine_every=a.refine_every, reset_every=a.reset_every, pose_opt=a.pose_opt, pose_lr=a.pose_lr)
Trainer(scene, cfg, a.out).train()
