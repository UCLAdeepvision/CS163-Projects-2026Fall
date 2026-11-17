"""Image-quality metrics (PSNR / SSIM / LPIPS) and the SSIM training loss."""
import torch
import torch.nn.functional as F


def _gauss_window(size=11, sigma=1.5, device="cuda"):
    x = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim_fn(a, b, size=11, return_map=False):
    """SSIM between two [h,w,3] images in 0..1 (returns a scalar tensor, or the per-pixel [h,w] map if return_map)."""
    a = a.permute(2, 0, 1)[None]; b = b.permute(2, 0, 1)[None]
    w = _gauss_window(size, 1.5, a.device).expand(3, 1, size, size)
    pad = size // 2
    mu_a = F.conv2d(a, w, padding=pad, groups=3); mu_b = F.conv2d(b, w, padding=pad, groups=3)
    s_aa = F.conv2d(a * a, w, padding=pad, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, w, padding=pad, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, w, padding=pad, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m[0].mean(0) if return_map else m.mean()


def ssim_loss_fn(pred, gt):
    return 1.0 - ssim_fn(pred, gt)


class Evaluator:
    """Callable: (pred [h,w,3], gt [h,w,3], mask [h,w] bool or None) in 0..1 -> dict(psnr, ssim, lpips).

    With a mask (True = valid pixel) ALL three metrics are evaluated over the unmasked pixels only:
      PSNR  : mean squared error over valid pixels;
      SSIM  : the per-pixel SSIM map averaged over valid pixels;
      LPIPS : the per-pixel LPIPS map (spatial=True) averaged over valid pixels.
    The caller first replaces the masked pixels of `pred` by the ground truth, so the SSIM/LPIPS windows that straddle the mask
    boundary see identical content on both sides.  Without a mask the usual whole-image metrics are returned."""

    def __init__(self, device="cuda"):
        import lpips
        self.lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.lps = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, pred, gt, mask=None):
        x, y = pred.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1
        if mask is not None and bool(mask.any()):
            mse = ((pred - gt) ** 2).mean(-1)[mask].mean().clamp(min=1e-10)
            s = ssim_fn(pred, gt, return_map=True)[mask].mean()
            lp = self.lps(x, y)[0, 0][mask].mean()
        else:
            mse = F.mse_loss(pred, gt).clamp(min=1e-10)
            s = ssim_fn(pred, gt)
            lp = self.lp(x, y)
        return {"psnr": (-10 * torch.log10(mse)).item(), "ssim": s.item(), "lpips": lp.item()}
