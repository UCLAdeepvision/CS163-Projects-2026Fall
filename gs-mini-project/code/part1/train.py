"""Part 1: plain 3D Gaussian Splatting on the Mip-NeRF 360 'garden' scene.

    python part1/train.py --out runs/p1_garden            # 7000 steps
    python part1/train.py --steps 30000 --out runs/p1_30k # the paper's schedule
Outputs in --out:  gaussians.pt, metrics.json (PSNR/SSIM/LPIPS on held-out views), renders/val/*.jpg
(ground truth | render), splat.ply (open it at https://superspl.at/editor).
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsmini import paths as _paths
from gsmini import Config, Scene, Trainer, export_ply

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=_paths.data_dir("part1_garden"))
ap.add_argument("--steps", type=int, default=7000)
ap.add_argument("--sh-degree", type=int, default=3, help="spherical-harmonics degree of the colour (0 = view-independent colour)")
ap.add_argument("--downscale", type=int, default=1, help="shrink images by this factor (use 2 on small GPUs / Colab)")
ap.add_argument("--lambda-ssim", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

scene = Scene(a.data, downscale=a.downscale)
cfg = Config(steps=a.steps, sh_degree=a.sh_degree, lambda_ssim=a.lambda_ssim, seed=a.seed,
             train_split="train", eval_splits=["val"], eval_every=max(a.steps // 4, 1000))
tr = Trainer(scene, cfg, a.out)
tr.train()
export_ply(tr.g.params, os.path.join(a.out, "splat.ply"))
