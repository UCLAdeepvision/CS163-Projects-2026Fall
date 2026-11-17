"""Part 4 goal: your model must reach an LPIPS of at most 0.58 on val_pass2 (the second drive, front camera).
LPIPS measures how different a render looks from the photo, so LOWER is better. (The baseline gets about 0.62.)

    python part4/check_goal.py --mine runs/p4_mine
(--baseline runs/p4_base is optional: it only prints the baseline next to yours.)"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gsmini.report import load_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--mine", nargs="+", required=True); ap.add_argument("--baseline", nargs="+", default=[])
ap.add_argument("--split", default="val_pass2"); ap.add_argument("--need", type=float, default=0.58)
a = ap.parse_args()


def vals(runs):
    out = []
    for r in runs:
        m = load_metrics(r)
        if m is None or a.split not in m:
            sys.exit(f"{r}: no finished run found (metrics.json with {a.split})")
        out.append(m[a.split]["lpips"]); print(f"  {r:40s} {a.split} LPIPS {out[-1]:.3f}  (PSNR {m[a.split]['psnr']:.2f})")
    return np.array(out)


if a.baseline:
    print("baseline:"); b = vals(a.baseline)
print("yours:"); y = vals(a.mine)
print(f"\nyours {y.mean():.3f}   (needed: at most {a.need:.2f}; lower LPIPS is better)" + (f"   baseline {b.mean():.3f}" if a.baseline else ""))
ok = y.mean() <= a.need
print("GOAL REACHED" if ok else "goal not reached yet")
sys.exit(0 if ok else 1)
