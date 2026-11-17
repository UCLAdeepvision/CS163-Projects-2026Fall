"""A small, readable 3D Gaussian Splatting trainer built on `gsplat`.

You are not expected to modify this file in parts 1-2; parts 2-4 plug extra
losses into `Trainer` through the `extra_loss` callback (see part*/train.py).

Pipeline per step:  pick a train view -> rasterize -> L1 + SSIM (+ your extra losses)
                    -> backward -> Adam step -> gsplat densification (split/clone/prune).
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from scipy.spatial import cKDTree

from .camopt import CamOpt
from .metrics import Evaluator, ssim_loss_fn


def rgb_to_sh0(rgb):
    return (rgb - 0.5) / 0.28209479177387814


def sh0_to_rgb(sh):
    return sh * 0.28209479177387814 + 0.5


class Gaussians:
    """Parameters of all splats (a dict of leaf tensors that gsplat's strategy edits in place)."""

    def __init__(self, xyz, rgb, sh_degree, init_opacity=0.1, device="cuda"):
        n = len(xyz)
        # initial scale = mean distance to the 3 nearest neighbours (as in the original 3DGS)
        d, _ = cKDTree(xyz).query(xyz, k=4)
        dist = np.clip(d[:, 1:].mean(1), 1e-4, None)
        scales = torch.from_numpy(np.log(dist))[:, None].repeat(1, 3).float()
        k = (sh_degree + 1) ** 2
        sh = torch.zeros(n, k, 3)
        sh[:, 0] = rgb_to_sh0(torch.from_numpy(rgb).float())
        self.params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.from_numpy(xyz).float()),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(F.normalize(torch.randn(n, 4), dim=-1)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), init_opacity))),
            "sh0": torch.nn.Parameter(sh[:, :1].contiguous()),
            "shN": torch.nn.Parameter(sh[:, 1:].contiguous()),
        }).to(device)

    def __len__(self):
        return self.params["means"].shape[0]

    # convenient activated views
    @property
    def means(self): return self.params["means"]
    @property
    def scales(self): return torch.exp(self.params["scales"])
    @property
    def quats(self): return F.normalize(self.params["quats"], dim=-1)
    @property
    def opacities(self): return torch.sigmoid(self.params["opacities"])
    @property
    def sh(self): return torch.cat([self.params["sh0"], self.params["shN"]], 1)

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.params.items()}


def render(g, w2c, K, w, h, sh_degree, mode="RGB+D"):
    """Differentiable render of the splats from one camera.

    Returns dict(rgb [h,w,3], depth [h,w], alpha [h,w], info).
    `depth` is gsplat's ACCUMULATED depth (render mode "D": the sum over Gaussians of weight * camera-space z); to get the depth of the visible
    surface divide it by `alpha` (that is gsplat's "ED", expected depth).
    """
    out, alpha, info = rasterization(
        means=g.means, quats=g.quats, scales=g.scales, opacities=g.opacities,
        colors=g.sh, viewmats=w2c[None], Ks=K[None], width=w, height=h,
        sh_degree=sh_degree, render_mode=mode, packed=False,
    )
    return dict(rgb=out[0, ..., :3].clamp(0, 1), depth=out[0, ..., 3], alpha=alpha[0, ..., 0], info=info)


class Config:
    def __init__(self, **kw):
        self.steps = 7000
        self.sh_degree = 3
        self.sh_every = 1000           # raise the SH degree by one every this many steps
        self.lambda_ssim = 0.2
        self.train_split = "train"
        self.eval_splits = ["val"]
        self.downscale = 1
        self.refine_stop = 0.5         # stop densifying at this fraction of training
        self.seed = 0
        self.eval_every = 0            # 0 = only at the end
        self.max_gaussians = 6_000_000
        self.grow_grad2d = 0.0002      # densification threshold (lower = more Gaussians)
        self.reset_every = 3000        # reset all opacities to a low value every this many steps (a huge number = never)
        self.refine_every = 100        # check for Gaussians to split / duplicate / prune every this many steps
        self.grow_scale3d = 0.01       # a Gaussian above this size (relative to the scene) is split, below it duplicated
        self.prune_scale3d = 0.1       # Gaussians larger than this (relative to the scene) are pruned; only active after the first opacity reset
        self.pose_opt = False          # learn per-frame + per-camera pose corrections (see camopt.py)
        self.pose_lr = 1e-3
        self.app_lr = 1e-3
        self.eval_pose_refine = 0      # >0: ALSO report metrics after this many steps of test-time pose refinement of each val view
        self.__dict__.update(kw)


class Trainer:
    def __init__(self, scene, cfg, out_dir, extra_loss=None, device="cuda", app_factory=None):
        """app_factory(n_train_frames, camera_index_per_frame, n_cameras) -> nn.Module with forward(k, rgb), forward_eval(cam, rgb), reg()  (optional appearance model)"""
        self.scene, self.cfg, self.out, self.dev = scene, cfg, out_dir, device
        os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.extra_loss = extra_loss
        self.train_ids = scene.indices(cfg.train_split)
        self.scene_scale = scene.scene_scale(cfg.train_split)
        init = getattr(scene, "init_points", None) or scene.points
        assert init is not None, "scene needs points.npz for initialisation"
        self.g = Gaussians(init["xyz"], init["rgb"].astype(np.float32) / 255.0,
                           cfg.sh_degree, device=device)
        s, steps = self.scene_scale, cfg.steps
        lrs = {"means": 1.6e-4 * s, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
               "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
        self.opts = {k: torch.optim.Adam([{"params": self.g.params[k], "lr": lr, "name": k}],
                                         eps=1e-15) for k, lr in lrs.items()}
        gamma = 0.01 ** (1.0 / steps)  # position lr decays to 1% over training
        self.means_sched = torch.optim.lr_scheduler.ExponentialLR(self.opts["means"], gamma=gamma)
        self.strategy = DefaultStrategy(
            verbose=False, refine_start_iter=500, refine_stop_iter=int(cfg.refine_stop * steps),
            refine_every=cfg.refine_every, reset_every=cfg.reset_every, prune_opa=0.005, grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d, prune_scale3d=cfg.prune_scale3d)
        self.strategy.check_sanity(self.g.params, self.opts)
        self.state = self.strategy.initialize_state(scene_scale=self.scene_scale)
        self.evaluator = Evaluator(device)
        self.log = []
        cams = sorted({f.camera for f in scene.frames}); self.cam_names = cams
        self.row = {i: k for k, i in enumerate(self.train_ids)}   # frame index -> row in the correction tables
        cidx = [cams.index(scene.frames[i].camera) for i in self.train_ids]
        self.camopt = CamOpt(len(self.train_ids), cidx, len(cams), device) if cfg.pose_opt else None
        self.appopt = app_factory(len(self.train_ids), cidx, len(cams)).to(device) if app_factory is not None else None
        self.aux_opts = [torch.optim.Adam(m.parameters(), lr=lr) for m, lr in
                         [(self.camopt, cfg.pose_lr), (self.appopt, cfg.app_lr)] if m is not None]

    # ------------------------------------------------------------------ training
    def train(self):
        cfg, sc = self.cfg, self.scene
        order, t0 = [], time.time()
        for step in range(cfg.steps):
            if not order:
                order = list(np.random.permutation(self.train_ids))
            i = int(order.pop())
            b = sc.get(i, self.dev)
            if self.camopt is not None:
                b["c2w"] = self.camopt(self.row[i], b["c2w"]); b["w2c"] = torch.linalg.inv(b["c2w"])
            sh_deg = min(step // cfg.sh_every, cfg.sh_degree)
            r = render(self.g, b["w2c"], b["K"], b["w"], b["h"], sh_deg)
            self.strategy.step_pre_backward(self.g.params, self.opts, self.state, step, r["info"])

            pred, gt, m = r["rgb"], b["image"], b["mask"]
            if self.appopt is not None:
                pred = self.appopt(self.row[i], pred)
            if m is not None:  # ignore masked pixels (people, sky, borders ...)
                pred, gt = pred * m[..., None], gt * m[..., None]
            l1 = (pred - gt).abs().mean()
            ssim_l = ssim_loss_fn(pred, gt)
            loss = (1 - cfg.lambda_ssim) * l1 + cfg.lambda_ssim * ssim_l
            logs = {"l1": l1.item()}
            if self.extra_loss is not None:
                ctx = dict(render=r, batch=b, scene=sc, step=step, steps=cfg.steps,
                           gaussians=self.g, scene_scale=self.scene_scale)
                extra, elog = self.extra_loss(ctx)
                if extra is not None:
                    loss = loss + extra
                    logs.update(elog)
            if self.camopt is not None:
                loss = loss + 1e-2 * self.camopt.reg()
            if self.appopt is not None:
                loss = loss + 1e-2 * self.appopt.reg()
            loss.backward()
            for o in list(self.opts.values()) + self.aux_opts:
                o.step(); o.zero_grad(set_to_none=True)
            self.means_sched.step()
            self.strategy.step_post_backward(self.g.params, self.opts, self.state, step, r["info"],
                                             packed=False)
            if len(self.g) > cfg.max_gaussians:  # safety valve against OOM
                self.strategy.refine_stop_iter = step
            if step % 200 == 0 or step == cfg.steps - 1:
                logs.update(step=step, n=len(self.g), t=round(time.time() - t0, 1))
                self.log.append(logs)
                print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in logs.items()), flush=True)
            if cfg.eval_every and (step + 1) % cfg.eval_every == 0 and step + 1 < cfg.steps:
                m = self.evaluate(save_images=False, max_views=20)
                print("  [eval]", json.dumps({k: {a: round(c, 3) for a, c in v.items()} for k, v in m.items()}))
        return self.finish()

    def finish(self):
        torch.save(self.g.state_dict(), os.path.join(self.out, "gaussians.pt"))
        if self.appopt is not None:   # the learned per-camera / per-frame colour corrections (students can plot them)
            torch.save(self.appopt.state_dict(), os.path.join(self.out, "appearance.pt"))
        metrics = self.evaluate(save_images=True)
        metrics["train_time_s"] = self.log[-1]["t"] if self.log else 0
        metrics["num_gaussians"] = len(self.g)
        metrics["steps"] = self.cfg.steps                  # the check scripts compare runs only if they trained for the same number of steps
        json.dump(metrics, open(os.path.join(self.out, "metrics.json"), "w"), indent=1)
        print("FINAL", json.dumps(metrics))
        return metrics

    # ---------------------------------------------------------------- evaluation
    def _cam_level(self, camera):
        """Corrections learned from TRAINING data that are shared by every frame of a physical camera."""
        ci = self.cam_names.index(camera) if camera in self.cam_names else None
        return ci

    def _eval_pose(self, b):
        """Apply the shared per-camera pose correction (learned on train views only) to a validation view."""
        ci = self._cam_level(b["camera"])
        if self.camopt is None or ci is None:
            return b["w2c"]
        from .camopt import so3_exp
        d = self.camopt.cam[ci].detach()
        T = torch.eye(4, device=self.dev); T[:3, :3] = so3_exp(d[:3]); T[:3, 3] = d[3:]
        return torch.linalg.inv(b["c2w"] @ T)

    def _refine_pose(self, b, steps):
        """Test-time pose refinement: optimise a 6-DoF correction of THIS view against its own image (Gaussians frozen).
        Uses the validation image, so it is reported separately (key '<split>+poseTTA')."""
        from .camopt import so3_exp
        d = torch.zeros(6, device=self.dev, requires_grad=True)
        opt = torch.optim.Adam([d], lr=2e-3)
        base = torch.linalg.inv(self._eval_pose(b))   # c2w incl. camera-level correction
        with torch.enable_grad():
            for _ in range(steps):
                T = torch.eye(4, device=self.dev)
                T = torch.cat([torch.cat([so3_exp(d[:3]), d[3:, None]], 1), T[3:]], 0)
                r = render(self.g, torch.linalg.inv(base @ T), b["K"], b["w"], b["h"], self.cfg.sh_degree)
                pred, gt = r["rgb"], b["image"]
                if b["mask"] is not None:
                    pred, gt = pred * b["mask"][..., None], gt * b["mask"][..., None]
                loss = 0.8 * (pred - gt).abs().mean() + 0.2 * ssim_loss_fn(pred, gt)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            T = torch.eye(4, device=self.dev)
            T = torch.cat([torch.cat([so3_exp(d[:3]), d[3:, None]], 1), T[3:]], 0)
            return torch.linalg.inv(base @ T)

    @torch.no_grad()
    def evaluate(self, save_images=True, max_views=None):
        res = {}
        for split in self.cfg.eval_splits:
            ids = self.scene.indices(split)
            if max_views and len(ids) > max_views:
                ids = ids[:: math.ceil(len(ids) / max_views)]
            rows, rows_tta = [], []
            for i in ids:
                b = self.scene.get(i, self.dev)
                w2c = self._eval_pose(b)
                r = render(self.g, w2c, b["K"], b["w"], b["h"], self.cfg.sh_degree)
                pred, gt = r["rgb"], b["image"]
                ci = self._cam_level(b["camera"])
                if self.appopt is not None and ci is not None:   # camera-level colour correction learned from TRAIN views only
                    pred = self.appopt.forward_eval(ci, pred).clamp(0, 1)
                if b["mask"] is not None:  # masked pixels do not count
                    pred = torch.where(b["mask"][..., None], pred, gt)
                rows.append(self.evaluator(pred, gt, b["mask"]))
                if self.cfg.eval_pose_refine and not max_views:
                    w2c2 = self._refine_pose(b, self.cfg.eval_pose_refine)
                    r2 = render(self.g, w2c2, b["K"], b["w"], b["h"], self.cfg.sh_degree)
                    p2 = r2["rgb"]
                    if self.appopt is not None and ci is not None:
                        p2 = self.appopt.forward_eval(ci, p2).clamp(0, 1)
                    if b["mask"] is not None:
                        p2 = torch.where(b["mask"][..., None], p2, gt)
                    rows_tta.append(self.evaluator(p2, gt, b["mask"]))
                if save_images:
                    self.save_triplet(split, b, r)
            res[split] = {k: float(np.mean([r_[k] for r_ in rows])) for k in rows[0]} | {"n_views": len(rows)}
            if rows_tta:
                res[split + "+poseTTA"] = {k: float(np.mean([r_[k] for r_ in rows_tta])) for k in rows_tta[0]} | {"n_views": len(rows_tta)}
        return res

    def save_triplet(self, split, b, r):
        from PIL import Image
        d = os.path.join(self.out, "renders", split)
        os.makedirs(d, exist_ok=True)
        gt = (b["image"].cpu().numpy() * 255).astype(np.uint8)
        pr = (r["rgb"].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(np.concatenate([gt, pr], 1)).save(os.path.join(d, b["name"] + ".jpg"), quality=90)


def export_ply(params, path):
    """Writes a standard 3DGS .ply (viewable in e.g. https://superspl.at/editor)."""
    from plyfile import PlyData, PlyElement
    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    n = len(p["means"])
    f_rest = p["shN"].transpose(0, 2, 1).reshape(n, -1)
    cols = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + \
           [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity", "scale_0", "scale_1", "scale_2",
                                                            "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([p["means"], np.zeros((n, 3)), p["sh0"][:, 0], f_rest, p["opacities"][:, None],
                           p["scales"], p["quats"]], 1).astype(np.float32)
    el = np.empty(n, dtype=[(c, "f4") for c in cols])
    for i, c in enumerate(cols):
        el[c] = data[:, i]
    PlyData([PlyElement.describe(el, "vertex")]).write(path)
