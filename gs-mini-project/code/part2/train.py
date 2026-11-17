"""Part 2: sparse-view 3DGS (12 views of an indoor room) with a monocular-depth prior.

    python part2/gen_depth.py                          # once: Depth Anything V2 on every view
    python part2/train.py --loss none  --out runs/p2_baseline
    python part2/train.py --loss l1    --w 1.0 --out runs/p2_l1
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsmini import paths as _paths, friendly as _friendly; _friendly.install()
sys.path.insert(0, os.path.dirname(__file__))
from gsmini import Config, Scene, Trainer
import depth_losses as DL

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=_paths.data_dir("part2_room"))
ap.add_argument("--depth", default="depth_da2_large", help="sub-folder with per-view .npy disparity maps")
ap.add_argument("--loss", default="l1", choices=["none", "l1"])
ap.add_argument("--w", type=float, default=1.0, help="depth-loss weight at the start (decays to 10%%)")
ap.add_argument("--steps", type=int, default=7000)
ap.add_argument("--train-split", default="train")
ap.add_argument("--out", required=True)
ap.add_argument("--init", default="random", choices=["sfm", "random"], help="sfm: the few triangulated points only; random: plus uniformly random points in the camera bounds")
ap.add_argument("--n-random", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sh-degree", type=int, default=3)
a = ap.parse_args()

scene = Scene(a.data)
if a.init == "random":  # weak initialisation: sparse SfM points + random points inside the (padded) camera bounding box
    import numpy as np
    c = np.stack([scene.frames[i].c2w[:3, 3] for i in scene.indices(a.train_split)])
    lo, hi = c.min(0), c.max(0); mid, half = (lo + hi) / 2, (hi - lo) / 2 * 1.5 + 0.5
    rng = np.random.default_rng(a.seed)
    xyz = np.concatenate([scene.points["xyz"], (mid + (rng.random((a.n_random, 3)) * 2 - 1) * half).astype(np.float32)])
    rgb = np.concatenate([scene.points["rgb"], np.full((a.n_random, 3), 128, np.uint8)])
    scene.init_points = {"xyz": xyz, "rgb": rgb}
cfg = Config(steps=a.steps, train_split=a.train_split, eval_splits=["val"], seed=a.seed, sh_degree=a.sh_degree)

# ---- pre-compute the aligned depth targets once (only needed with --loss l1) ----
targets = {}
if a.loss == "l1":
    for i in scene.indices(a.train_split):
        disp = scene.load_aux(a.depth, i)
        sp = scene.sparse_depth(i)
        t = None if sp is None else DL.align_depth_to_sparse(disp, *sp)
        if t is not None:
            targets[i] = t
    print(f"aligned depth for {len(targets)}/{len(scene.indices(a.train_split))} views")


def extra_loss(ctx):
    if a.loss == "none":
        return None, {}
    b, r = ctx["batch"], ctx["render"]
    w = a.w * (0.1 + 0.9 * (1 - ctx["step"] / ctx["steps"]))
    loss, logs = 0.0, {}
    if b["idx"] in targets:
        l = DL.l1_inverse_depth_loss(r["depth"], r["alpha"], targets[b["idx"]], b["mask"])
        loss = loss + w * l; logs["d_l1"] = l.item()
    return loss, logs


Trainer(scene, cfg, a.out, extra_loss=extra_loss).train()
