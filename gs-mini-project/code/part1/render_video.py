"""Render a fly-through video from a trained model:  python part1/render_video.py --run runs/p1_garden
The camera follows the training-camera path (smoothly interpolated), so it is always inside the scene."""
import argparse, os, sys
import numpy as np, torch, imageio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsmini import paths as _paths
from scipy.spatial.transform import Rotation, Slerp
from gsmini import Gaussians, Scene, render

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", default=_paths.data_dir("part1_garden"))
ap.add_argument("--split", default="train")
ap.add_argument("--frames", type=int, default=240)
ap.add_argument("--sh-degree", type=int, default=3)
ap.add_argument("--downscale", type=int, default=1)
ap.add_argument("--out", default=None)
a = ap.parse_args()

scene = Scene(a.data, downscale=a.downscale)
p = torch.load(os.path.join(a.run, "gaussians.pt"))
g = Gaussians(np.random.rand(8, 3).astype(np.float32), np.zeros((8, 3), np.float32), a.sh_degree)
for k in g.params:
    g.params[k] = torch.nn.Parameter(p[k].cuda())
ids = scene.indices(a.split)
if a.split == "train":                          # order the cameras by angle around the scene centre
    c = np.stack([scene.frames[i].c2w[:3, 3] for i in ids]); m = c.mean(0)
    ids = [ids[k] for k in np.argsort(np.arctan2(c[:, 1] - m[1], c[:, 0] - m[0]))]
poses = np.stack([scene.frames[i].c2w for i in ids]); f0 = scene.frames[ids[0]]
t = np.linspace(0, len(ids) - 1, a.frames)
sl = Slerp(np.arange(len(ids)), Rotation.from_matrix(poses[:, :3, :3]))
R = sl(t).as_matrix(); T = np.stack([np.interp(t, np.arange(len(ids)), poses[:, k, 3]) for k in range(3)], 1)
out = a.out or os.path.join(a.run, "flythrough.mp4")
wr = imageio.get_writer(out, fps=30)
with torch.no_grad():
    for k in range(a.frames):
        c2w = np.eye(4, dtype=np.float32); c2w[:3, :3] = R[k]; c2w[:3, 3] = T[k]
        w2c = torch.linalg.inv(torch.from_numpy(c2w)).cuda()
        r = render(g, w2c, torch.from_numpy(f0.K).cuda(), f0.w, f0.h, a.sh_degree)
        wr.append_data((r["rgb"].cpu().numpy() * 255).astype(np.uint8))
wr.close(); print("wrote", out)
