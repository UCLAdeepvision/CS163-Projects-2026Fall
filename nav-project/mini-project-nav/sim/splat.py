"""Gaussian splat renderer in plain PyTorch (runs on CPU, Apple GPU or CUDA).

Reads a standard 3D Gaussian splatting .ply file and renders it from a pinhole camera, following the usual EWA splatting
recipe:
    1. project every Gaussian to a 2D ellipse on the image,
    2. sort the ellipses by depth,
    3. alpha-composite them front to back, one 16x16 pixel tile at a time.
Colors are the degree-0 spherical harmonics only, which is all these scenes carry.
"""
import numpy as np
import torch
from plyfile import PlyData

SH0 = 0.28209479177387814       # the constant that turns a degree-0 spherical harmonic coefficient into a color
BLUR = 0.3                      # px^2 added to the 2D covariance, so that no splat is thinner than a pixel
MIN_ALPHA = 1 / 255             # contributions below this are dropped
MAX_ALPHA = 0.99


def pick_device(name="auto"):
    """CUDA if there is one, otherwise the CPU. The Apple GPU has to be requested explicitly with name="mps"."""
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_ply(path):
    """Read a 3DGS .ply. Returns float32 arrays with one row per Gaussian:
    mean [N, 3], quat [N, 4] (unit, w x y z), scale [N, 3], opacity [N], color [N, 3] in 0..1."""
    v = PlyData.read(path)["vertex"].data
    columns = lambda names: np.stack([v[n] for n in names], 1).astype(np.float32)
    quat = columns([f"rot_{i}" for i in range(4)])
    return dict(mean=columns(["x", "y", "z"]),
                quat=quat / np.linalg.norm(quat, axis=1, keepdims=True),
                scale=np.exp(columns([f"scale_{i}" for i in range(3)])),                 # stored as log-scale
                opacity=1 / (1 + np.exp(-v["opacity"].astype(np.float32))),              # stored as a logit
                color=np.clip(columns([f"f_dc_{i}" for i in range(3)]) * SH0 + 0.5, 0, 1))


def covariance(quat, scale):
    """3D covariance R diag(s)^2 R^T of each Gaussian. Torch tensors in, [N, 3, 3] out; quat is (w, x, y, z)."""
    w, x, y, z = quat.unbind(-1)
    rot = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                       2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                       2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
    m = rot * scale[:, None, :]
    return m @ m.transpose(1, 2)


class Splats:
    """The Gaussians of one scene, held on `device`."""

    def __init__(self, path, device="auto", min_opacity=0.02):
        self.device = pick_device(device)
        g = {k: torch.from_numpy(a) for k, a in read_ply(path).items()}
        keep = g["opacity"] > min_opacity                    # nearly transparent Gaussians cost time and show nothing
        g = {k: a[keep] for k, a in g.items()}
        self.mean = g["mean"].to(self.device)
        self.cov = covariance(g["quat"], g["scale"]).to(self.device)     # in world coordinates
        self.opacity = g["opacity"].to(self.device)
        self.color = g["color"].to(self.device)

    @torch.no_grad()
    def render(self, c2w, width, height, fov_deg=90.0, near=0.05, far=1e10, tile=None):       # no far limit: the sky is made of Gaussians a few kilometers away
        """Render one view. `c2w` is the 4x4 camera-to-world matrix (OpenCV axes: x right, y down, z forward).
        Returns a uint8 array [height, width, 3]."""
        if tile is None:
            tile = 64 if self.device.type == "cuda" else 16        # large tiles are fastest on a GPU (5x), small ones on a CPU
        focal = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
        ellipses = self._project(c2w, width, height, focal, near, far)
        return self._composite(ellipses, width, height, tile)

    def _project(self, c2w, width, height, f, near, far):
        """Project the Gaussians that can be seen onto the image, sorted from near to far.

        Returns a dict of tensors, one entry per visible Gaussian: image position (u, v), pixel radius, the inverse 2D
        covariance (inv_a, inv_b, inv_c = the entries xx, xy, yy), opacity and color.
        """
        dev = self.device
        c2w = torch.as_tensor(c2w, dtype=torch.float32, device=dev)
        R, t = c2w[:3, :3], c2w[:3, 3]

        # points in camera coordinates; keep those in front of the camera and roughly inside the view
        p = (self.mean - t) @ R
        z = p[:, 2]
        ok = (z > near) & (z < far)
        ok &= (p[:, 0].abs() < z * (width / (2 * f)) * 1.5 + 1.0) & (p[:, 1].abs() < z * (height / (2 * f)) * 1.5 + 1.0)
        idx = ok.nonzero().squeeze(1)
        p, z = p[idx], z[idx]

        # 2D covariance: cov2d = J W cov3d W^T J^T, where J is the Jacobian of the perspective projection
        cam_cov = R.T @ self.cov[idx] @ R
        J = torch.zeros(len(idx), 2, 3, device=dev)
        J[:, 0, 0] = f / z
        J[:, 1, 1] = f / z
        J[:, 0, 2] = -f * p[:, 0] / z ** 2
        J[:, 1, 2] = -f * p[:, 1] / z ** 2
        cov2 = J @ cam_cov @ J.transpose(1, 2)
        a, b, c = cov2[:, 0, 0] + BLUR, cov2[:, 0, 1], cov2[:, 1, 1] + BLUR
        det = (a * c - b * b).clamp_min(1e-8)
        half = 0.5 * (a + c)
        radius = 3 * torch.sqrt(half + torch.sqrt((half * half - det).clamp_min(0)))    # 3 sigma along the long axis
        u = f * p[:, 0] / z + width / 2
        v = f * p[:, 1] / z + height / 2

        # drop ellipses that miss the image; a few degenerate splats right at the lens would otherwise cover the whole frame
        seen = (u + radius > 0) & (u - radius < width) & (v + radius > 0) & (v - radius < height)
        seen &= ~((radius > 150) & (z < 3.0))
        order = torch.argsort(z[seen])
        keep = idx[seen][order]
        u, v, radius, a, b, c, det = (q[seen][order] for q in (u, v, radius, a, b, c, det))
        return dict(u=u, v=v, radius=radius, inv_a=c / det, inv_b=-b / det, inv_c=a / det,
                    opacity=self.opacity[keep], color=self.color[keep])

    def _composite(self, e, width, height, tile):
        """Alpha-composite the sorted ellipses into an image, one tile at a time."""
        dev = self.device
        image = torch.zeros(height, width, 3, device=dev)
        for y0 in range(0, height, tile):
            for x0 in range(0, width, tile):
                x1, y1 = min(x0 + tile, width), min(y0 + tile, height)
                # the ellipses whose bounding square touches this tile (still sorted by depth)
                hit = ((e["u"] + e["radius"] >= x0) & (e["u"] - e["radius"] <= x1)
                       & (e["v"] + e["radius"] >= y0) & (e["v"] - e["radius"] <= y1)).nonzero().squeeze(1)
                if len(hit) == 0:
                    continue
                ys, xs = torch.meshgrid(torch.arange(y0, y1, device=dev) + 0.5, torch.arange(x0, x1, device=dev) + 0.5, indexing="ij")
                px, py = xs.reshape(-1, 1), ys.reshape(-1, 1)
                transmittance = torch.ones(px.shape[0], device=dev)          # how much light still gets through, per pixel
                color = torch.zeros(px.shape[0], 3, device=dev)
                for s in range(0, len(hit), 1024):                           # front to back in chunks
                    g = hit[s:s + 1024]
                    dx, dy = px - e["u"][g][None], py - e["v"][g][None]
                    power = -0.5 * (e["inv_a"][g] * dx * dx + 2 * e["inv_b"][g] * dx * dy + e["inv_c"][g] * dy * dy)
                    alpha = (e["opacity"][g] * torch.exp(power.clamp(max=0))).clamp(max=MAX_ALPHA)
                    alpha = torch.where(alpha > MIN_ALPHA, alpha, torch.zeros_like(alpha))
                    through = torch.cumprod(1 - alpha, dim=1)
                    before = transmittance[:, None] * torch.cat([torch.ones_like(through[:, :1]), through[:, :-1]], 1)
                    color += (alpha * before) @ e["color"][g]
                    transmittance = transmittance * through[:, -1]
                    if float(transmittance.max()) < 1e-3:                    # the tile is opaque, nothing behind it matters
                        break
                image[y0:y1, x0:x1] = color.view(y1 - y0, x1 - x0, 3)
        return (image.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
