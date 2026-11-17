"""Scene loading for the gsmini project. See Appendix 11.1 of the project PDF for the on-disk format."""
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class Frame:
    name: str          # file stem, e.g. "front_000012"
    camera: str        # physical camera ("front", "left", ...)
    K: np.ndarray      # [3,3] float32, for the (possibly downscaled) image
    c2w: np.ndarray    # [4,4] float32, OpenCV convention
    w: int
    h: int


class Scene:
    """All images of a scene (uint8, in RAM) + cameras + split lists.

    `downscale` shrinks images and intrinsics (e.g. 2 -> half resolution).
    Optional per-frame auxiliary data (depth maps, normals, ...) lives in
    `<root>/<kind>/<name>.npy` and is read with `load_aux`.
    """

    def __init__(self, root, downscale=1, load_masks=True):
        self.root = root
        meta = json.load(open(os.path.join(root, "cameras.json")))
        self.meta = meta.get("meta", {})
        self.splits = {k: list(v) for k, v in meta["splits"].items()}
        self.frames, self.images, self.masks = [], [], []
        self._aux = {}
        for f in meta["frames"]:
            img = Image.open(os.path.join(root, f["file"])).convert("RGB")
            w, h = img.size  # actual size on disk (should equal f["w"], f["h"])
            K = np.array(f["K"], dtype=np.float32)
            K[0] *= w / f["w"]; K[1] *= h / f["h"]
            if downscale != 1:
                w2, h2 = round(w / downscale), round(h / downscale)
                img = img.resize((w2, h2), Image.LANCZOS)
                K[0] *= w2 / w; K[1] *= h2 / h
                w, h = w2, h2
            name = os.path.splitext(os.path.basename(f["file"]))[0]
            self.frames.append(Frame(name, f.get("camera", "cam"), K,
                                     np.array(f["c2w"], dtype=np.float32), w, h))
            self.images.append(torch.from_numpy(np.asarray(img)).clone())  # [h,w,3] uint8
            m = None
            mp = os.path.join(root, "masks", name + ".png")
            if load_masks and os.path.exists(mp):
                mi = Image.open(mp).convert("L")
                if mi.size != (w, h):
                    mi = mi.resize((w, h), Image.NEAREST)
                m = torch.from_numpy(np.asarray(mi) > 127).clone()  # [h,w] bool, True = valid
            self.masks.append(m)
        pts = os.path.join(root, "points.npz")
        self.points = dict(np.load(pts)) if os.path.exists(pts) else None

    def __len__(self):
        return len(self.frames)

    def indices(self, split):
        return self.splits[split]

    # -- tensors for one frame --------------------------------------------------
    def get(self, i, device="cuda"):
        """Returns dict: image [h,w,3] float 0..1, mask [h,w] bool or None, K [3,3], c2w [4,4], w2c [4,4]."""
        f = self.frames[i]
        c2w = torch.from_numpy(f.c2w).to(device)
        m = self.masks[i]
        return dict(
            idx=i, name=f.name, camera=f.camera, w=f.w, h=f.h,
            image=self.images[i].to(device).float() / 255.0,
            mask=None if m is None else m.to(device),
            K=torch.from_numpy(f.K).to(device),
            c2w=c2w, w2c=torch.linalg.inv(c2w),
        )

    # -- auxiliary per-frame data -----------------------------------------------
    def aux_path(self, kind, i):
        return os.path.join(self.root, kind, self.frames[i].name + ".npy")

    def has_aux(self, kind, i):
        return os.path.exists(self.aux_path(kind, i))

    def load_aux(self, kind, i, device="cuda"):
        """Loads <root>/<kind>/<name>.npy as a float32 tensor (cached on CPU)."""
        key = (kind, i)
        if key not in self._aux:
            self._aux[key] = torch.from_numpy(np.load(self.aux_path(kind, i)).astype(np.float32))
        return self._aux[key].to(device)

    # -- misc -------------------------------------------------------------------
    def scene_scale(self, split="train"):
        """Radius of the camera set (used to scale learning rates / densification)."""
        c = np.stack([self.frames[i].c2w[:3, 3] for i in self.splits[split]])
        return float(np.linalg.norm(c - c.mean(0), axis=1).max() * 1.1)

    def sparse_depth(self, i):
        """Per-view sparse SfM depth, if points.npz carries tracks (part 2).
        Returns (xy [M,2] pixel coords in the stored image, z [M] camera-space depth), or None.
        """
        p = self.points
        if p is None or "track_frames" not in p:
            return None
        sel = p["track_frames"] == i
        if not sel.any():
            return None
        pid = p["track_points"][sel]
        xy = p["track_xy"][sel]
        f = self.frames[i]
        w2c = np.linalg.inv(f.c2w)
        X = p["xyz"][pid]
        z = (w2c[:3, :3] @ X.T).T[:, 2] + w2c[2, 3]
        # track_xy was stored at the original resolution; rescale to the loaded one
        xy = xy * np.array([f.w / p["track_wh"][0], f.h / p["track_wh"][1]], dtype=np.float32)
        return xy.astype(np.float32), z.astype(np.float32)
