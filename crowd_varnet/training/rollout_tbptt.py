"""全 episode rollout + TBPTT 训练循环 + 纯 forward val loss。

返回字典形式的损失分项，便于按通道（rho / vx / vy / speed）观察训练动态。
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from ..models.cost import density_support_mask, masked_mean_sq
from ..models.varnet import CrowdVarNet


_CH_NAMES = ("rho", "vx", "vy", "speed")


def _format_components(comp: Dict[str, float]) -> str:
    """把分项打成单行可读字符串：rho/vx/vy/speed 的 MSE + 加权贡献。"""
    return (
        f"rho={comp['rho']:.4f} vx={comp['vx']:.4f} "
        f"vy={comp['vy']:.4f} sp={comp['speed']:.4f} | "
        f"w*: rho={comp['rho_w']:.4f} vx={comp['vx_w']:.4f} "
        f"vy={comp['vy_w']:.4f} sp={comp['speed_w']:.4f}"
    )


def _per_channel_stats(
    err: torch.Tensor, dm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (ch_sq_sum[4], n_mask_pixels[scalar])。

    err: [B,4,H,W] 平方误差；dm: [B,1,H,W] 0/1 掩码。
    每个通道用同一个 dm，分母为 dm 像素总数。
    """
    with torch.no_grad():
        ch_sq_sum = (err * dm).sum(dim=(0, 2, 3))  # [4]
        n_mask = dm.sum()
    return ch_sq_sum, n_mask


def _rho_bg_stats(
    err: torch.Tensor, x_hat: torch.Tensor, dm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """密度通道在 GT 背景（dm==0）上的诊断：返回 (sum_se_bg, sum_abs_bg, n_bg)。"""
    with torch.no_grad():
        bg = 1.0 - dm  # [B,1,H,W]
        rho_err2 = err[:, 0:1]
        rho_hat = x_hat[:, 0:1]
        sum_se_bg = (rho_err2 * bg).sum()
        sum_abs_bg = (rho_hat.abs() * bg).sum()
        n_bg = bg.sum()
    return sum_se_bg, sum_abs_bg, n_bg


def rollout_tbptt_epoch(
    model: CrowdVarNet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    warmup: int = 5,
    k_prime: int = 8,
    grad_clip: float = 0.5,
    log_interval: int = 1,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
    after_step_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, float]:
    """一个 epoch 的 TBPTT rollout 训练。

    返回 dict，键：
      ``total``                  当前训练目标（与旧版本一致：通道加权后的 masked-MSE）
      ``rho/vx/vy/speed``        各通道未加权 MSE
      ``rho_w/vx_w/vy_w/speed_w`` 各通道加权贡献（= ch_w[c] * 通道 MSE / 4）
    """
    model.train()
    ch_w = model.cost_fn.ch_w  # [1,4,1,1]
    ch_w_flat = ch_w.view(-1).detach()  # [4]

    # epoch-level accumulators
    ep_loss_sum = 0.0
    ep_steps = 0
    ep_ch_sq = torch.zeros(4, device=device)
    ep_ch_n = torch.zeros((), device=device)
    ep_rho_bg_se = torch.zeros((), device=device)
    ep_rho_bg_abs = torch.zeros((), device=device)
    ep_rho_bg_n = torch.zeros((), device=device)

    eprefix = (
        f"[{int(epoch) + 1}/{int(num_epochs)}] "
        if epoch is not None and num_epochs is not None
        else ""
    )
    n_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        x_gt_seq, obs_seq, mask_seq = batch
        x_gt_seq = x_gt_seq.to(device)
        obs_seq = obs_seq.to(device)
        mask_seq = mask_seq.to(device)

        B, L, C, H, W = x_gt_seq.shape
        assert L > warmup, f"episode too short: L={L}, warmup={warmup}"

        history_buf = x_gt_seq[:, :warmup].clone()

        # batch-level accumulators (for logging)
        b_loss_sum = 0.0
        b_steps = 0
        b_ch_sq = torch.zeros(4, device=device)
        b_ch_n = torch.zeros((), device=device)
        b_rho_bg_se = torch.zeros((), device=device)
        b_rho_bg_abs = torch.zeros((), device=device)
        b_rho_bg_n = torch.zeros((), device=device)
        n_windows = 0

        for window_start in range(warmup, L, k_prime):
            window_end = min(window_start + k_prime, L)
            window_loss = 0.0
            window_steps = 0
            for t in range(window_start, window_end):
                x_hat = model.forward(history_buf, obs_seq[:, t], mask_seq[:, t])
                err = (x_hat - x_gt_seq[:, t]).pow(2)
                dm = density_support_mask(x_gt_seq[:, t], model.cost_fn.rho_mask_thr)
                # density: penalize everywhere (suppress background leakage);
                # velocity: only where GT has pedestrians (matches compute_loss)
                full = torch.ones_like(dm)
                m4 = torch.cat([full, dm, dm, dm], dim=1)
                step_loss = masked_mean_sq(err * ch_w, m4)
                window_loss = window_loss + step_loss
                window_steps += 1

                ch_sq, n_mask = _per_channel_stats(err, dm)
                b_ch_sq += ch_sq
                b_ch_n += n_mask
                bg_se, bg_abs, bg_n = _rho_bg_stats(err, x_hat, dm)
                b_rho_bg_se += bg_se
                b_rho_bg_abs += bg_abs
                b_rho_bg_n += bg_n
                b_steps += 1

                history_buf = torch.cat([history_buf[:, 1:], x_hat.unsqueeze(1)], dim=1)

            window_loss = window_loss / max(window_steps, 1)
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            optimizer.step()
            if after_step_callback is not None:
                after_step_callback()

            b_loss_sum += float(window_loss.item()) * window_steps
            n_windows += 1
            history_buf = history_buf.detach()

        # batch summary
        batch_total = b_loss_sum / max(b_steps, 1)
        ep_loss_sum += b_loss_sum
        ep_steps += b_steps
        ep_ch_sq += b_ch_sq
        ep_ch_n += b_ch_n
        ep_rho_bg_se += b_rho_bg_se
        ep_rho_bg_abs += b_rho_bg_abs
        ep_rho_bg_n += b_rho_bg_n

        if log_interval > 0 and (
            (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == n_batches
        ):
            n_safe = b_ch_n.clamp_min(1.0)
            b_ch_mse = (b_ch_sq / n_safe).detach().cpu().tolist()
            b_components = {
                _CH_NAMES[i]: float(b_ch_mse[i]) for i in range(4)
            }
            for i, name in enumerate(_CH_NAMES):
                b_components[f"{name}_w"] = float(
                    ch_w_flat[i].item() * b_ch_mse[i] / 4.0
                )
            n_bg_safe = b_rho_bg_n.clamp_min(1.0)
            b_rho_bg_mse = float((b_rho_bg_se / n_bg_safe).item())
            b_rho_bg_mean = float((b_rho_bg_abs / n_bg_safe).item())
            print(
                f"{eprefix}ep-batch {batch_idx + 1}/{n_batches}  "
                f"windows={n_windows}  loss(avg)={batch_total:.6f}  "
                f"{_format_components(b_components)}  "
                f"| rho_bg_mse={b_rho_bg_mse:.6f} rho_bg_mean={b_rho_bg_mean:.6f}",
                flush=True,
            )

    n_safe = ep_ch_n.clamp_min(1.0)
    ep_ch_mse = (ep_ch_sq / n_safe).detach().cpu().tolist()
    out: Dict[str, float] = {
        "total": ep_loss_sum / max(ep_steps, 1),
    }
    for i, name in enumerate(_CH_NAMES):
        out[name] = float(ep_ch_mse[i])
        out[f"{name}_w"] = float(ch_w_flat[i].item() * ep_ch_mse[i] / 4.0)
    n_bg_safe = ep_rho_bg_n.clamp_min(1.0)
    out["rho_bg_mse"] = float((ep_rho_bg_se / n_bg_safe).item())
    out["rho_bg_mean"] = float((ep_rho_bg_abs / n_bg_safe).item())
    return out


def rollout_val_loss(
    model: CrowdVarNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    warmup: int = 5,
) -> Dict[str, float]:
    """纯 forward rollout 验证：返回与训练对齐的分项 dict（``total`` + 4 通道 MSE/加权贡献）。

    注意：solver 内部用 ``torch.autograd.grad`` 算更新方向，**必须**在 grad-enabled 模式下跑，
    所以这里不用 ``@torch.no_grad``；通过对 ``x_hat`` 立即 ``detach`` 阻止图累积，
    并对模型参数用 ``requires_grad=False`` 防止参数梯度累计内存。
    """
    model.eval()
    ch_w = model.cost_fn.ch_w
    ch_w_flat = ch_w.view(-1).detach()

    loss_sum = 0.0
    steps = 0
    ch_sq_sum = torch.zeros(4, device=device)
    n_mask_sum = torch.zeros((), device=device)
    rho_bg_se = torch.zeros((), device=device)
    rho_bg_abs = torch.zeros((), device=device)
    rho_bg_n = torch.zeros((), device=device)

    saved_grad_states = [(p, p.requires_grad) for p in model.parameters() if p.requires_grad]
    for p, _ in saved_grad_states:
        p.requires_grad_(False)
    try:
        for batch in loader:
            x_gt_seq, obs_seq, mask_seq = batch
            x_gt_seq = x_gt_seq.to(device)
            obs_seq = obs_seq.to(device)
            mask_seq = mask_seq.to(device)

            B, L, C, H, W = x_gt_seq.shape
            history_buf = x_gt_seq[:, :warmup].clone()
            for t in range(warmup, L):
                x_hat = model.forward(history_buf, obs_seq[:, t], mask_seq[:, t]).detach()
                err = (x_hat - x_gt_seq[:, t]).pow(2)
                dm = density_support_mask(x_gt_seq[:, t], model.cost_fn.rho_mask_thr)
                full = torch.ones_like(dm)
                m4 = torch.cat([full, dm, dm, dm], dim=1)
                step_loss = masked_mean_sq(err * ch_w, m4)
                loss_sum += float(step_loss.item())
                steps += 1
                ch_sq, n_mask = _per_channel_stats(err, dm)
                ch_sq_sum += ch_sq
                n_mask_sum += n_mask
                bg_se, bg_abs, bg_n = _rho_bg_stats(err, x_hat, dm)
                rho_bg_se += bg_se
                rho_bg_abs += bg_abs
                rho_bg_n += bg_n
                history_buf = torch.cat([history_buf[:, 1:], x_hat.unsqueeze(1)], dim=1)
    finally:
        for p, was_req in saved_grad_states:
            p.requires_grad_(was_req)

    n_safe = n_mask_sum.clamp_min(1.0)
    ch_mse = (ch_sq_sum / n_safe).detach().cpu().tolist()
    out: Dict[str, float] = {
        "total": loss_sum / max(steps, 1),
    }
    for i, name in enumerate(_CH_NAMES):
        out[name] = float(ch_mse[i])
        out[f"{name}_w"] = float(ch_w_flat[i].item() * ch_mse[i] / 4.0)
    n_bg_safe = rho_bg_n.clamp_min(1.0)
    out["rho_bg_mse"] = float((rho_bg_se / n_bg_safe).item())
    out["rho_bg_mean"] = float((rho_bg_abs / n_bg_safe).item())
    return out
