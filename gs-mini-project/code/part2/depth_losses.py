"""Part 2: turning a monocular depth foundation model into a training signal.

Depth Anything V2 gives *relative inverse depth* ("disparity"): up to an unknown scale and shift per
image.  The 3DGS renderer gives *metric-ish depth* in the units of the COLMAP reconstruction.
`fit_scale_shift` and `align_depth_to_sparse` (the glue between the two) are PROVIDED: read them.  You implement `l1_inverse_depth_loss`.
"""
import numpy as np
import torch


def fit_scale_shift(pred, target, iters=3):
    """PROVIDED.  Least-squares fit  target ~= a * pred + b  (robust: re-fit after dropping the worst 20% residuals).

    pred, target: 1-D float tensors/arrays of equal length (>= 2 points).  Returns (a, b) as floats.
    """
    pred = torch.as_tensor(pred, dtype=torch.float64); target = torch.as_tensor(target, dtype=torch.float64)
    keep = torch.ones_like(pred, dtype=torch.bool)
    a, b = 1.0, 0.0
    for _ in range(iters):
        A = torch.stack([pred[keep], torch.ones_like(pred[keep])], 1)
        sol = torch.linalg.lstsq(A, target[keep, None]).solution[:, 0]
        a, b = sol[0].item(), sol[1].item()
        res = (a * pred + b - target).abs()
        keep = res <= torch.quantile(res, 0.8)
    return a, b


def align_depth_to_sparse(disp, xy, z):
    """PROVIDED.  Align a relative-inverse-depth map to sparse SfM depth.

    disp: [H,W] float tensor, the foundation model's output (bigger = closer, arbitrary scale/shift).
    xy:   [M,2] pixel coordinates (x right, y down) of sparse SfM points visible in this view.
    z:    [M]   their camera-space depth (same units as the renderer's depth).
    Returns a [H,W] tensor of *inverse depth in SfM units*:  a * disp + b  fitted so that it matches 1/z
    at the sparse points.  (Fit in inverse-depth space: that is where the network's error is closest to
    an affine transform.)  Return None if there are fewer than 8 usable points.
    """
    H, W = disp.shape
    xy = torch.as_tensor(xy, device=disp.device); z = torch.as_tensor(z, device=disp.device)
    ok = (z > 1e-3) & (xy[:, 0] >= 0) & (xy[:, 0] <= W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= H - 1)
    if ok.sum() < 8:
        return None
    xy, z = xy[ok], z[ok]
    d = disp[xy[:, 1].round().long(), xy[:, 0].round().long()]
    a, b = fit_scale_shift(d, 1.0 / z)
    return a * disp + b


def l1_inverse_depth_loss(render_depth, alpha, target_inv_depth, mask=None):
    """L1 between rendered inverse depth and the aligned monocular inverse depth.

    render_depth: [H,W] accumulated depth from the splatting renderer: the sum over Gaussians of (blending weight x depth), NOT yet
                  divided by the opacity.  Dividing by `alpha` (= the sum of the weights) gives the depth of the visible surface.
                  Background: https://docs.gsplat.studio/main/apis/rasterization.html  (see "Depth Rendering": mode "D" is the accumulated
                  depth, mode "ED" the expected depth = D / alpha).
    alpha:        [H,W] accumulated opacity
    target_inv_depth: [H,W] output of align_depth_to_sparse
    mask: optional [H,W] bool, True = use this pixel.

    Hint: there are several divisions in this function; clamping (`tensor.clamp(min=...)`) avoids dividing by 0.
    """
    # TODO: write your implementation here (see the docstring above)
    raise NotImplementedError("TODO")
