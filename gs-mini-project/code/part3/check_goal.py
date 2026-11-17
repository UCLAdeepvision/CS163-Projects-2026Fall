"""Part 3 goal: your best run must be at least +0.75 dB PSNR above the baseline on val_offtraj.

    python part3/check_goal.py --baseline runs/p3_base --mine runs/p3_best

The baseline must have been trained for the DEFAULT number of steps (a baseline trained for another number of steps is a different baseline);
otherwise this script refuses to compare it. Your best run may use any number of steps (--steps)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gsmini.report import load_metrics

DEFAULT_STEPS = 7000                                             # the default of part3/train.py --steps
ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True); ap.add_argument("--mine", required=True)
ap.add_argument("--split", default="val_offtraj"); ap.add_argument("--need", type=float, default=0.75)
a = ap.parse_args()


def load(run, must_be_default):
    m = load_metrics(run)
    if m is None or a.split not in m:
        sys.exit(f"{run}: no finished run found (metrics.json with {a.split})")
    steps = m.get("steps")
    if steps is None:
        sys.exit(f"{run}: this run did not record its number of steps. Train it again.")
    if must_be_default and steps != DEFAULT_STEPS:
        sys.exit(f"{run}: trained for {steps} steps, but the baseline must use the default {DEFAULT_STEPS}. Train it again without --steps.")
    print(f"  {run:40s} {a.split} PSNR {m[a.split]['psnr']:.2f}  LPIPS {m[a.split]['lpips']:.3f}  ({steps} steps)")
    return m[a.split]["psnr"]


print("baseline:"); b = load(a.baseline, True)
print("yours:"); y = load(a.mine, False)
gain = y - b
print(f"\nbaseline {b:.2f} dB   yours {y:.2f} dB   gain {gain:+.2f} dB   (needed: {a.need:+.2f})")
print("GOAL REACHED" if gain >= a.need else "goal not reached yet")
sys.exit(0 if gain >= a.need else 1)
