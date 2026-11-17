"""Run a depth foundation model on the training views (and, for analysis, the validation views).
    python part2/gen_depth.py --size large
Writes data/part2_room/<kind>/<frame>.npy  (float16, relative inverse depth, bigger = closer)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsmini import paths as _paths
from gsmini.depth import run_on_scene

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=_paths.data_dir("part2_room"))
ap.add_argument("--size", default="large", help="small | base | large (Depth Anything V2) or any HF model id")
ap.add_argument("--splits", nargs="+", default=["train"])
ap.add_argument("--flip-aug", action="store_true")
a = ap.parse_args()
run_on_scene(a.data, f"depth_da2_{a.size}", a.size, a.splits, a.flip_aug)
