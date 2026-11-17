"""Optional learnable pose corrections that make 3DGS robust to imperfect capture (used by Part 3 through --pose-opt).

* CamOpt  - a small SE(3) correction per training frame plus one shared correction per physical camera.
            Handheld/robot rigs have per-camera extrinsic offsets and timing jitter of ~1 degree; 1 degree is ~7 pixels
            at a 400 px focal length, which is enough to blur a reconstruction.  The shared per-camera term learns the
            constant (extrinsic) part quickly; the per-frame term absorbs the jitter.
"""
import torch


def so3_exp(w):
    """axis-angle [3] -> rotation matrix [3,3] (differentiable)."""
    z = torch.zeros((), device=w.device, dtype=w.dtype)
    K = torch.stack([z, -w[2], w[1], w[2], z, -w[0], -w[1], w[0], z]).reshape(3, 3)
    return torch.linalg.matrix_exp(K)


class CamOpt(torch.nn.Module):
    def __init__(self, n_frames, cam_index, n_cams, device="cuda"):
        super().__init__()
        self.frame = torch.nn.Parameter(torch.zeros(n_frames, 6, device=device))
        self.cam = torch.nn.Parameter(torch.zeros(n_cams, 6, device=device))
        self.cam_index = torch.as_tensor(cam_index, device=device)

    def forward(self, k, c2w):
        """k: index of the frame in the optimiser's table, c2w [4,4] -> corrected c2w (correction applied in camera frame)."""
        d = self.frame[k] + self.cam[self.cam_index[k]]
        T = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
        T = torch.cat([torch.cat([so3_exp(d[:3]), d[3:, None]], 1), T[3:]], 0)
        return c2w @ T

    def reg(self):
        return (self.frame ** 2).mean()
