"""Sanity tests for your l1_inverse_depth_loss in part2/depth_losses.py.   python part2/check_losses.py"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gsmini import friendly; friendly.install()
import depth_losses as DL

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

torch.manual_seed(0)
H, W = 40, 60
z_true = 2.0 + 0.05 * torch.arange(W).float()[None].repeat(H, 1)        # depth ramp

alpha = torch.ones(H, W)
inv_t = 1 / z_true
check("l1 loss is 0 at the target", DL.l1_inverse_depth_loss(z_true, alpha, inv_t).item() < 1e-5)
check("l1 loss divides depth by alpha", DL.l1_inverse_depth_loss(z_true * 0.5, torch.full((H, W), 0.5), inv_t).item() < 1e-5)
m = torch.zeros(H, W, dtype=torch.bool); m[:, :30] = True
bad = z_true.clone(); bad[:, 30:] *= 3
check("l1 loss respects the mask", DL.l1_inverse_depth_loss(bad, alpha, inv_t, m).item() < 1e-5)
print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED"); sys.exit(0 if ok else 1)
