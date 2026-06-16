"""python -m crowd_varnet → smoke test."""
import torch
import torch.nn as nn

from .models import CrowdVarNet


def run_smoke() -> None:
    device = torch.device("cpu")
    B, T, H, W = 2, 5, 36, 12

    class DummyPedPred(nn.Module):
        def forward(self, inp, hidden=None, *, horizon=1):
            t = inp if torch.is_tensor(inp) else torch.as_tensor(inp)
            if t.dim() == 5:
                t = t[:, -1]
            return t.unsqueeze(1)

    model = CrowdVarNet(
        DummyPedPred().to(device),
        freeze_phi=True,
        T_hist=T,
        n_iter=4,
        solver_hidden=16,
    ).to(device)

    history = torch.rand(B, T, 4, H, W, device=device)
    obs_mask = torch.zeros(B, 1, H, W, device=device)
    obs_mask[:, :, :, : W // 2] = 1.0
    obs = torch.rand(B, 4, H, W, device=device) * obs_mask
    x_gt = torch.rand(B, 4, H, W, device=device)

    loss, info = model.compute_loss(history, obs, obs_mask, x_gt)
    print(f"loss={loss.item():.4f}  phi_mse={info['phi_mse']:.4f}  recon={info['recon']:.4f}")
    print("Smoke test passed.")


if __name__ == "__main__":
    run_smoke()
