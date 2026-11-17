"""Monocular depth with a foundation model (Depth Anything V2, via Hugging Face `transformers`).

The standard models predict *relative inverse depth* (disparity): larger = closer, with an unknown
per-image scale and shift. The "metric_*" models predict depth in metres directly.
"""
import os

import numpy as np
import torch
from PIL import Image

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    # METRIC variants: output is depth in metres (not inverse depth); trained on outdoor / indoor data
    "metric_outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "metric_indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


class DepthModel:
    def __init__(self, size="large", device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        name = MODELS.get(size, size)
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.net = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
        self.dev = device

    @torch.no_grad()
    def __call__(self, pil_img):
        """PIL RGB image -> float32 numpy [h,w] relative inverse depth at the image's resolution."""
        w, h = pil_img.size
        inp = self.proc(images=pil_img, return_tensors="pt").to(self.dev)
        out = self.net(**inp).predicted_depth[None]  # [1,1,H',W']
        out = torch.nn.functional.interpolate(out, size=(h, w), mode="bicubic", align_corners=False)
        return out[0, 0].float().cpu().numpy()


def run_on_scene(root, kind, size="large", split_names=None, flip_aug=False):
    """Writes <root>/<kind>/<frame>.npy for every frame in the given splits (default: all frames)."""
    import json
    cj = json.load(open(os.path.join(root, "cameras.json")))
    ids = sorted({i for s in (split_names or cj["splits"]) for i in cj["splits"][s]})
    os.makedirs(os.path.join(root, kind), exist_ok=True)
    m = DepthModel(size)
    for n, i in enumerate(ids):
        f = cj["frames"][i]["file"]
        stem = os.path.splitext(os.path.basename(f))[0]
        d = m(Image.open(os.path.join(root, f)).convert("RGB"))
        if flip_aug:  # average with the prediction on the mirrored image (a cheap test-time augmentation)
            d2 = m(Image.open(os.path.join(root, f)).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT))[:, ::-1]
            d = 0.5 * (d + d2)
        np.save(os.path.join(root, kind, stem + ".npy"), d.astype(np.float16))
        if n % 20 == 0:
            print(f"depth {n}/{len(ids)}", flush=True)
