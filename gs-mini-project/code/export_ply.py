"""Turn a trained run into a .ply file that you can open in a viewer (for example https://superspl.at/editor).

    python export_ply.py --run runs/p3_best                  # writes runs/p3_best/splat.ply
    python export_ply.py --run runs/p3_best --out my_scene.ply

It reads the run's gaussians.pt (every part's train.py writes it), so it works for any part and for your own scans."""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsmini import export_ply

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="the run folder (the --out of a train.py command)")
ap.add_argument("--out", default=None, help="the .ply file to write (default: <run>/splat.ply)")
a = ap.parse_args()
ckpt = os.path.join(a.run, "gaussians.pt")
if not os.path.exists(ckpt):
    sys.exit(f"{ckpt} not found: --run must be a run folder with a finished training run (it has gaussians.pt)")
out = a.out or os.path.join(a.run, "splat.ply")
export_ply(torch.load(ckpt, map_location="cpu"), out)
print(f"wrote {out} ({os.path.getsize(out) / 2**20:.0f} MB): drag it into https://superspl.at/editor")
