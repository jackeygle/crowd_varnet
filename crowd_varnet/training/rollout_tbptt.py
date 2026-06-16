"""全 episode rollout + TBPTT 训练循环 + 纯 forward val loss。

核心 loss 由 4 部分组成（可独立开关）：
  1. recon_loss        — 通道加权 + 密度支撑掩码的 MSE（基础重建损失）
  2. bg_suppress_loss  — 背景密度抑制（lambda_bg > 0 启用，优化 1）
  3. dir_loss          — 速度方向 cosine 损失（lambda_dir > 0 启用，优化 4）
  4. lookahead_weight  — 多步 lookahead 加权（gamma > 1.0 启用，优化 2）
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from ..models.cost import density_support_mask, masked_mean_sq, topk_mean_sq
from ..models.varnet import CrowdVarNet


def _to_pure_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert any tensor subclass (e.g. GridData) to a plain torch.Tensor."""
    if type(x) is torch.Tensor:
        return x
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    out.copy_(x)
    return out


_CH_NAMES = ("rho", "vx", "vy", "speed")


def _format_components(comp: Dict[str, float]) -> str:
    return (
        f"rho={comp['rho']:.4f} vx={comp['vx']:.4f} "
        f"vy={comp['vy']:.4f} sp={comp['speed']:.4f} | "
        f"w*: rho={comp['rho_w']:.4f} vx={comp['vx_w']:.4f} "
        f"vy={comp['vy_w']:.4f} sp={comp['speed_w']:.4f}"
    )


def _per_channel_stats(
    err: torch.Tensor, dm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        ch_sq_sum = (err * dm).sum(dim=(0, 2, 3))
        n_mask = dm.sum()
    return ch_sq_sum, n_mask


def _rho_bg_stats(
    err: torch.Tensor, x_hat: torch.Tensor, dm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        bg = 1.0 - dm
        rho_err2 = err[:, 0:1]
        rho_hat = x_hat[:, 0:1]
        sum_se_bg = (rho_err2 * bg).sum()
        sum_abs_bg = (rho_hat.abs() * bg).sum()
        n_bg = bg.sum()
    return sum_se_bg, sum_abs_bg, n_bg


def _bg_suppress_term(x_hat: torch.Tensor, dm: torch.Tensor) -> torch.Tensor:
    """优化 1: 背景密度抑制。

    在 GT 背景区（dm=0），鼓励预测密度趋向 0（用 |rho_hat| 的 mean）。
    避免模型在背景区"铺底"小密度。
    """
    bg = 1.0 - dm  # [B, 1, H, W]
    rho_hat = x_hat[:, 0:1]
    return (rho_hat.abs() * bg).sum() / bg.sum().clamp_min(1.0)


def _direction_loss(
    x_hat: torch.Tensor, x_gt: torch.Tensor, dm: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """优化 4: 速度方向 cosine 损失。

    在 GT 有人 (dm=1) 且 GT 速度非零的格点上，惩罚预测速度方向与 GT 偏差。
    返回 mean(1 - cos_sim)，越小越好（0 = 方向完全一致）。
    """
    vel_pred = x_hat[:, 1:3]  # [B, 2, H, W]
    vel_gt = x_gt[:, 1:3]
    mag_pred = vel_pred.pow(2).sum(1, keepdim=True).clamp_min(eps).sqrt()
    mag_gt = vel_gt.pow(2).sum(1, keepdim=True).clamp_min(eps).sqrt()
    cos_sim = (vel_pred * vel_gt).sum(1, keepdim=True) / (mag_pred * mag_gt)
    # 只在有人且 GT 速度非零的地方算
    valid = dm * (mag_gt > eps).float()
    loss = ((1.0 - cos_sim) * valid).sum() / valid.sum().clamp_min(1.0)
    return loss


def _mass_conservation_loss(x_hat: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    """优化 5 (新): 质量守恒。

    每帧的总密度（积分）应该接近 GT 的总密度。
    用相对误差 |sum(pred) - sum(gt)| / max(sum(gt), 1) 让损失尺度无关。
    """
    total_pred = x_hat[:, 0:1].sum(dim=(2, 3))   # [B, 1]
    total_gt = x_gt[:, 0:1].sum(dim=(2, 3))
    rel_err = (total_pred - total_gt).abs() / total_gt.clamp_min(1.0)
    return rel_err.mean()


def _log_density_mse(x_hat: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    """优化 6 (新): log-space 密度 MSE。

    用 log(1+ρ) 比较预测和 GT。在小值区域更敏感（0 vs 0.01 比 1 vs 1.01 更重要）。
    密度通道做的，所有像素都参与（全图）。
    """
    log_pred = torch.log1p(x_hat[:, 0:1].clamp_min(0.0))
    log_gt = torch.log1p(x_gt[:, 0:1].clamp_min(0.0))
    return (log_pred - log_gt).pow(2).mean()


def _velocity_sparsity_loss(x_hat: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """优化 7 (新): 速度场稀疏正则。

    在 GT 没人 (density < thr) 的地方，惩罚预测的速度幅度。
    防止 attention 把速度信息扩散到本来无人的区域。
    """
    vx = x_hat[:, 1:2]
    vy = x_hat[:, 2:3]
    speed_sq = vx.pow(2) + vy.pow(2)
    bg = (x_gt[:, 0:1] < 0.05).float()  # GT 背景区（没人）
    return (speed_sq * bg).sum() / (bg.sum() + eps)


def _density_only_loss(x_hat: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    """优化 8 (新): 密度通道独立 loss（全图 MSE）。

    专门强化密度学习，与主 loss 分离。
    """
    return (x_hat[:, 0:1] - x_gt[:, 0:1]).pow(2).mean()


def _heteroscedastic_nll(
    x_hat: torch.Tensor,
    log_var: torch.Tensor,
    x_gt: torch.Tensor,
    ch_w: torch.Tensor,
    m4: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """异方差 NLLL（Kendall & Gal 2017）应用于前 3 通道（ρ, vx, vy）。

    nll_pixel = 0.5 * exp(-log_var) * (pred - gt)^2 + 0.5 * log_var

    返回 (loss, sigma_mean_for_logging)。
    """
    err3 = (x_hat[:, :3] - x_gt[:, :3]).pow(2)
    inv_var = torch.exp(-log_var)  # log_var clamped in model; safe.
    nll_pix = 0.5 * inv_var * err3 + 0.5 * log_var
    # ch_w[:, :3] has shape [1, 3, 1, 1]; m4[:, :3] has shape [B, 3, H, W]
    weighted = nll_pix * ch_w[:, :3]
    loss = masked_mean_sq(weighted, m4[:, :3])
    sigma_mean = torch.exp(0.5 * log_var).mean().detach()
    return loss, sigma_mean


# ----------------------------------------------------------------------
# Teacher-style losses (PedPred3 NLLL recipe; see docs/loss_alignment.md)
# ----------------------------------------------------------------------

def _poisson_nll_density(
    rho_hat: torch.Tensor, rho_gt: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Poisson NLL on density channel, full-image average.

    NLL = ρ̂ - ρ_gt * log(ρ̂),  ρ̂ clamped to ≥ eps for numerical safety.
    Matches PedPred3's "mean NLLL density" metric.
    """
    rho_safe = rho_hat.clamp_min(eps)
    nll = rho_safe - rho_gt * torch.log(rho_safe)
    return nll.mean()


def _gaussian_nll_velocity_mean(
    v_hat: torch.Tensor, v_gt: torch.Tensor, rho_gt: torch.Tensor
) -> torch.Tensor:
    """ρ_gt-weighted Gaussian NLL on velocity (mean only, σ=1).

    NLL_pix = 0.5 * (v_hat - v_gt)²,  weighted by ρ_gt across pixels.
    Matches PedPred3's "mean weighted NLLL vel_est" (without the unc term).
    """
    err = 0.5 * (v_hat - v_gt).pow(2).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    w = rho_gt.clamp_min(0.0)
    den = w.sum().clamp_min(1e-6)
    return (err * w).sum() / den


def _gaussian_nll_velocity_full(
    v_hat: torch.Tensor,
    v_gt: torch.Tensor,
    log_var: torch.Tensor,
    rho_gt: torch.Tensor,
) -> torch.Tensor:
    """Full Gaussian NLL on velocity (mean + variance), ρ_gt-weighted.

    NLL_pix = 0.5 * log(σ̂²) + 0.5 * (v_hat - v_gt)² / σ̂²,
    sum over (vx, vy) channels, weighted by ρ_gt.
    Matches PedPred3's "vel_est + vel_unc".

    log_var: shape [B, 2, H, W] (one per velocity channel) or
             [B, 1, H, W] (shared) — broadcast handled.
    """
    err = (v_hat - v_gt).pow(2)
    inv_var = torch.exp(-log_var)
    # 0.5 * log_var + 0.5 * inv_var * err per channel, then sum over (vx, vy)
    nll = 0.5 * log_var + 0.5 * inv_var * err
    nll_sum = nll.sum(dim=1, keepdim=True)  # [B, 1, H, W]
    w = rho_gt.clamp_min(0.0)
    den = w.sum().clamp_min(1e-6)
    return (nll_sum * w).sum() / den


def _teacher_style_recon(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    log_var: Optional[torch.Tensor],
    lambda_vel: float = 1.0,
    lambda_rho: float = 1.0,
    lambda_vx: float = 1.0,
    lambda_vy: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-style recon = heteroscedastic NLLL on all 3 channels (ρ, vx, vy).

    Density: full-image heteroscedastic Gaussian NLLL, weighted by lambda_rho.
    Velocity: ρ_gt-weighted heteroscedastic Gaussian NLLL, with per-channel
    weights lambda_vx / lambda_vy.

    If ``log_var`` is None: falls back to MSE (σ=1 equivalent).
    If ``log_var`` is provided [B, 3, H, W]: full het-NLLL on all channels.

    Returns (loss, sigma_mean_for_logging).
    """
    rho_hat = x_hat[:, 0:1]
    rho_gt = x_gt[:, 0:1]
    v_hat = x_hat[:, 1:3]
    v_gt = x_gt[:, 1:3]

    if log_var is not None:
        # log_var: [B, 3, H, W] — channels 0=ρ, 1=vx, 2=vy
        log_var_rho = log_var[:, 0:1]  # [B, 1, H, W]
        log_var_vx = log_var[:, 1:2]   # [B, 1, H, W]
        log_var_vy = log_var[:, 2:3]   # [B, 1, H, W]

        # Density: full-image het-NLLL
        err_rho = (rho_hat - rho_gt).pow(2)
        nll_rho = 0.5 * torch.exp(-log_var_rho) * err_rho + 0.5 * log_var_rho
        loss_rho = nll_rho.mean()

        # Velocity vx: ρ_gt-weighted het-NLLL
        err_vx = (v_hat[:, 0:1] - v_gt[:, 0:1]).pow(2)
        nll_vx = 0.5 * torch.exp(-log_var_vx) * err_vx + 0.5 * log_var_vx
        w = rho_gt.clamp_min(0.0)
        den = w.sum().clamp_min(1e-6)
        loss_vx = (nll_vx * w).sum() / den

        # Velocity vy: ρ_gt-weighted het-NLLL
        err_vy = (v_hat[:, 1:2] - v_gt[:, 1:2]).pow(2)
        nll_vy = 0.5 * torch.exp(-log_var_vy) * err_vy + 0.5 * log_var_vy
        loss_vy = (nll_vy * w).sum() / den

        sigma_mean = torch.exp(0.5 * log_var).mean().detach()
    else:
        # Fallback: MSE-style (σ=1)
        loss_rho = (rho_hat - rho_gt).pow(2).mean()
        w = rho_gt.clamp_min(0.0)
        den = w.sum().clamp_min(1e-6)
        err_vx = 0.5 * (v_hat[:, 0:1] - v_gt[:, 0:1]).pow(2)
        loss_vx = (err_vx * w).sum() / den
        err_vy = 0.5 * (v_hat[:, 1:2] - v_gt[:, 1:2]).pow(2)
        loss_vy = (err_vy * w).sum() / den
        sigma_mean = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

    return lambda_rho * loss_rho + lambda_vx * loss_vx + lambda_vy * loss_vy, sigma_mean


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
    # === Loss optimizations (all default to disabled = original behavior) ===
    lambda_bg: float = 0.0,        # 优化 1: 背景密度抑制权重 (建议 0.5-2.0)
    lookahead_gamma: float = 1.0,  # 优化 2: 多步 lookahead 增长率 (1.0 = 无, 1.1 = 后期步加权)
    lambda_dir: float = 0.0,       # 优化 4: 速度方向损失权重 (建议 0.1-0.5)
    lambda_mass: float = 0.0,      # 优化 5: 质量守恒损失权重 (建议 0.1-0.5)
    lambda_log_rho: float = 0.0,   # 优化 6: log-space 密度 MSE 权重 (建议 0.3-1.0)
    weight_mode: str = "hard_mask",  # "hard_mask" (default, current) | "continuous" (target_density)
    loss_style: str = "mse",       # "mse" (default) | "teacher" (Poisson density + ρ_gt-weighted Gaussian vel NLLL)
    teacher_lambda_vel: float = 1.0,  # weight on velocity NLLL when loss_style=teacher
    teacher_lambda_rho: float = 1.0,  # weight on density NLLL when loss_style=teacher
    teacher_lambda_vx: float = 1.0,   # weight on vx NLLL when loss_style=teacher
    teacher_lambda_vy: float = 1.0,   # weight on vy NLLL when loss_style=teacher
    unobs_loss_weight: float = 1.0,  # 未观测区有人处的 loss 权重倍数 (1.0=不加权，5.0=5x 强调)
    sched_sampling_prob: float = 0.0,  # 概率：用 GT 替换 history_buf 里的自估帧 (0=纯 rollout, 0.3=30% 帧用 GT)
    lambda_vel_sparsity: float = 0.0,  # 速度场稀疏正则 (0=关闭, 0.5-2.0 推荐)
    lambda_density: float = 0.0,  # 密度通道独立 MSE loss (0=关闭, 1.0-3.0 推荐)
    topk_percent: float = 1.0,  # OHEM-style: only top-k% largest errors contribute to loss
                                # (1.0 = standard mean, 0.2 = top 20% hardest pixels). Default disabled.
) -> Dict[str, float]:
    """一个 epoch 的 TBPTT rollout 训练。

    新增可选优化（用 lambda_*/gamma=1.0 关闭等价旧行为）：
      lambda_bg > 0      ：每步 loss 加 lambda_bg * |rho_bg|.mean()
      lookahead_gamma>1.0：窗口内第 t 步权重 = gamma**t（远期步更重要）
      lambda_dir > 0     ：每步 loss 加 lambda_dir * (1 - cos(v_pred, v_gt))
      lambda_mass > 0    ：每步 loss 加 lambda_mass * |sum(rho_pred) - sum(rho_gt)| / sum(rho_gt)
      lambda_log_rho > 0 ：每步 loss 加 lambda_log_rho * MSE(log(1+rho_pred), log(1+rho_gt))
    """
    model.train()
    ch_w = model.cost_fn.ch_w
    ch_w_flat = ch_w.view(-1).detach()
    use_nlll = bool(getattr(model, "predict_uncertainty", False))

    ep_loss_sum = 0.0
    ep_recon_sum = 0.0
    ep_bg_sum = 0.0
    ep_dir_sum = 0.0
    ep_mass_sum = 0.0
    ep_log_rho_sum = 0.0
    ep_sigma_sum = 0.0
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
        x_gt_seq = _to_pure_tensor(x_gt_seq).to(device)
        obs_seq = _to_pure_tensor(obs_seq).to(device)
        mask_seq = _to_pure_tensor(mask_seq).to(device)

        B, L, C, H, W = x_gt_seq.shape
        assert L > warmup, f"episode too short: L={L}, warmup={warmup}"

        history_buf = x_gt_seq[:, :warmup].clone()

        b_loss_sum = 0.0
        b_recon_sum = 0.0
        b_bg_sum = 0.0
        b_dir_sum = 0.0
        b_mass_sum = 0.0
        b_log_rho_sum = 0.0
        b_sigma_sum = 0.0
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
            window_weight_sum = 0.0
            window_steps = 0

            for t_idx, t in enumerate(range(window_start, window_end)):
                if use_nlll:
                    x_hat, log_var = model.forward_with_var(
                        history_buf, obs_seq[:, t], mask_seq[:, t]
                    )
                else:
                    x_hat = model.forward(history_buf, obs_seq[:, t], mask_seq[:, t])
                    log_var = None
                x_gt = x_gt_seq[:, t]
                err = (x_hat - x_gt).pow(2)
                dm = density_support_mask(x_gt, model.cost_fn.rho_mask_thr)

                # === 1. 基础重建 loss ===
                full = torch.ones_like(dm)
                if weight_mode == "continuous":
                    # 他们风格：速度通道用连续 target_density 加权（不是二值掩码）
                    # 密度通道全图，不加权
                    target_rho = x_gt[:, 0:1].clamp_min(0.0)
                    m4 = torch.cat([full, target_rho, target_rho, target_rho], dim=1)
                elif weight_mode == "full":
                    # 全图所有通道都算 loss（鼓励未观测区也准）
                    m4 = torch.cat([full, full, full, full], dim=1)
                else:
                    # hard mask（默认，原有行为）
                    m4 = torch.cat([full, dm, dm, dm], dim=1)

                # === 1b. 未观测区加权（防止 loss 被背景稀释）===
                if unobs_loss_weight != 1.0:
                    obs_m = mask_seq[:, t]  # [B, 1, H, W]
                    unobs_m = 1.0 - obs_m
                    # 未观测区 + 有人处加权 (unobs_loss_weight)
                    # 其他地方权重 1.0
                    boost = 1.0 + (unobs_loss_weight - 1.0) * unobs_m * dm
                    m4 = m4 * boost

                if use_nlll and log_var is not None:
                    if loss_style == "teacher":
                        # Teacher-style: het-NLLL on all 3 channels (ρ, vx, vy)
                        step_recon, sigma_mean = _teacher_style_recon(
                            x_hat, x_gt, log_var=log_var,
                            lambda_vel=teacher_lambda_vel,
                            lambda_rho=teacher_lambda_rho,
                            lambda_vx=teacher_lambda_vx,
                            lambda_vy=teacher_lambda_vy,
                        )
                    elif loss_style == "detached_nll":
                        # MSE drives solver (same as baseline) + detached NLLL drives var_head only
                        step_recon_mse = topk_mean_sq(err * ch_w, m4, topk_percent)
                        err_det = (x_hat.detach() - x_gt).pow(2)[:, :3]
                        nll_aux = 0.5 * torch.exp(-log_var) * err_det + 0.5 * log_var
                        step_recon = step_recon_mse + 0.1 * nll_aux.mean()
                        sigma_mean = torch.exp(0.5 * log_var).mean().detach()
                    else:
                        # Default: heteroscedastic NLLL on first 3 channels (ρ, vx, vy)
                        step_recon, sigma_mean = _heteroscedastic_nll(
                            x_hat, log_var, x_gt, ch_w, m4
                        )
                else:
                    if loss_style == "teacher":
                        step_recon, sigma_mean = _teacher_style_recon(
                            x_hat, x_gt, log_var=None, lambda_vel=teacher_lambda_vel,
                            lambda_rho=teacher_lambda_rho,
                            lambda_vx=teacher_lambda_vx,
                            lambda_vy=teacher_lambda_vy,
                        )
                    else:
                        # mse or detached_nll without var_head → pure MSE
                        step_recon = topk_mean_sq(err * ch_w, m4, topk_percent)
                        sigma_mean = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

                # === 2. 背景密度抑制 (优化 1) ===
                if lambda_bg > 0.0:
                    step_bg = _bg_suppress_term(x_hat, dm)
                else:
                    step_bg = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

                # === 3. 速度方向 loss (优化 4) ===
                if lambda_dir > 0.0:
                    step_dir = _direction_loss(x_hat, x_gt, dm)
                else:
                    step_dir = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

                # === 4. 质量守恒 (优化 5) ===
                if lambda_mass > 0.0:
                    step_mass = _mass_conservation_loss(x_hat, x_gt)
                else:
                    step_mass = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

                # === 5. log-space 密度 MSE (优化 6) ===
                if lambda_log_rho > 0.0:
                    step_log_rho = _log_density_mse(x_hat, x_gt)
                else:
                    step_log_rho = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

                # === 7. 速度场稀疏正则 (新)===
                if lambda_vel_sparsity > 0.0:
                    step_vel_sparsity = _velocity_sparsity_loss(x_hat, x_gt)
                else:
                    step_vel_sparsity = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

                # === 8. 密度独立 loss (新)===
                if lambda_density > 0.0:
                    step_density = _density_only_loss(x_hat, x_gt)
                else:
                    step_density = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

                # 单步 total loss
                step_loss = (
                    step_recon
                    + lambda_bg * step_bg
                    + lambda_dir * step_dir
                    + lambda_mass * step_mass
                    + lambda_log_rho * step_log_rho
                    + lambda_vel_sparsity * step_vel_sparsity
                    + lambda_density * step_density
                )

                # === 4. 多步 lookahead 加权 (优化 2) ===
                # 窗口内第 t_idx 步权重 = gamma**t_idx，远期步更重要
                step_weight = float(lookahead_gamma) ** t_idx
                window_loss = window_loss + step_weight * step_loss
                window_weight_sum += step_weight

                # 累计统计
                with torch.no_grad():
                    b_recon_sum += float(step_recon.item())
                    b_bg_sum += float(step_bg.item())
                    b_dir_sum += float(step_dir.item())
                    b_mass_sum += float(step_mass.item())
                    b_log_rho_sum += float(step_log_rho.item())
                    if use_nlll:
                        b_sigma_sum += float(sigma_mean.item())
                window_steps += 1

                ch_sq, n_mask = _per_channel_stats(err, dm)
                b_ch_sq += ch_sq
                b_ch_n += n_mask
                bg_se, bg_abs, bg_n = _rho_bg_stats(err, x_hat, dm)
                b_rho_bg_se += bg_se
                b_rho_bg_abs += bg_abs
                b_rho_bg_n += bg_n
                b_steps += 1

                # === 6. Scheduled sampling on history ===
                # With prob `sched_sampling_prob`, replace self-estimate with GT
                # in the history buffer. Gives the model "GT anchors" to prevent
                # rollout drift in unobserved regions.
                # Inference is always pure self-estimated (no GT access).
                if sched_sampling_prob > 0.0 and torch.rand(()).item() < sched_sampling_prob:
                    history_push = x_gt.detach()  # use GT
                else:
                    history_push = x_hat
                history_buf = torch.cat([history_buf[:, 1:], history_push.unsqueeze(1)], dim=1)

            # 用加权平均（如果 lookahead_gamma=1.0 等价于普通平均）
            window_loss = window_loss / max(window_weight_sum, 1e-6)
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

        batch_total = b_loss_sum / max(b_steps, 1)
        ep_loss_sum += b_loss_sum
        ep_recon_sum += b_recon_sum
        ep_bg_sum += b_bg_sum
        ep_dir_sum += b_dir_sum
        ep_mass_sum += b_mass_sum
        ep_log_rho_sum += b_log_rho_sum
        ep_sigma_sum += b_sigma_sum
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
            extras = []
            if lambda_bg > 0:
                extras.append(f"bg={b_bg_sum / max(b_steps, 1):.4f}")
            if lambda_dir > 0:
                extras.append(f"dir={b_dir_sum / max(b_steps, 1):.4f}")
            if lambda_mass > 0:
                extras.append(f"mass={b_mass_sum / max(b_steps, 1):.4f}")
            if lambda_log_rho > 0:
                extras.append(f"logrho={b_log_rho_sum / max(b_steps, 1):.4f}")
            if use_nlll:
                extras.append(f"sigma={b_sigma_sum / max(b_steps, 1):.4f}")
            extras_str = (" " + " ".join(extras)) if extras else ""
            print(
                f"{eprefix}ep-batch {batch_idx + 1}/{n_batches}  "
                f"windows={n_windows}  loss(avg)={batch_total:.6f}  "
                f"recon={b_recon_sum / max(b_steps, 1):.4f}{extras_str}  "
                f"{_format_components(b_components)}  "
                f"| rho_bg_mse={b_rho_bg_mse:.6f} rho_bg_mean={b_rho_bg_mean:.6f}",
                flush=True,
            )

    n_safe = ep_ch_n.clamp_min(1.0)
    ep_ch_mse = (ep_ch_sq / n_safe).detach().cpu().tolist()
    out: Dict[str, float] = {
        "total": ep_loss_sum / max(ep_steps, 1),
        "recon": ep_recon_sum / max(ep_steps, 1),
        "bg_suppress": ep_bg_sum / max(ep_steps, 1),
        "dir": ep_dir_sum / max(ep_steps, 1),
        "mass": ep_mass_sum / max(ep_steps, 1),
        "log_rho": ep_log_rho_sum / max(ep_steps, 1),
        "sigma_mean": ep_sigma_sum / max(ep_steps, 1) if use_nlll else 0.0,
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
    lambda_bg: float = 0.0,
    lambda_dir: float = 0.0,
    lambda_mass: float = 0.0,
    lambda_log_rho: float = 0.0,
    weight_mode: str = "hard_mask",
    loss_style: str = "mse",
    teacher_lambda_vel: float = 1.0,
    teacher_lambda_rho: float = 1.0,
    teacher_lambda_vx: float = 1.0,
    teacher_lambda_vy: float = 1.0,
) -> Dict[str, float]:
    """纯 forward rollout 验证。可选地在 val 上也算辅助 loss 用于诊断。"""
    model.eval()
    ch_w = model.cost_fn.ch_w
    ch_w_flat = ch_w.view(-1).detach()
    use_nlll = bool(getattr(model, "predict_uncertainty", False))

    loss_sum = 0.0
    recon_sum = 0.0
    bg_sum = 0.0
    dir_sum = 0.0
    mass_sum = 0.0
    log_rho_sum = 0.0
    sigma_sum = 0.0
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
            x_gt_seq = _to_pure_tensor(x_gt_seq).to(device)
            obs_seq = _to_pure_tensor(obs_seq).to(device)
            mask_seq = _to_pure_tensor(mask_seq).to(device)

            B, L, C, H, W = x_gt_seq.shape
            history_buf = x_gt_seq[:, :warmup].clone()
            for t in range(warmup, L):
                if use_nlll:
                    x_hat_t, log_var_t = model.forward_with_var(
                        history_buf, obs_seq[:, t], mask_seq[:, t]
                    )
                    x_hat = x_hat_t.detach()
                    log_var = log_var_t.detach() if log_var_t is not None else None
                else:
                    x_hat = model.forward(history_buf, obs_seq[:, t], mask_seq[:, t]).detach()
                    log_var = None
                x_gt = x_gt_seq[:, t]
                err = (x_hat - x_gt).pow(2)
                dm = density_support_mask(x_gt, model.cost_fn.rho_mask_thr)

                full = torch.ones_like(dm)
                if weight_mode == "continuous":
                    target_rho = x_gt[:, 0:1].clamp_min(0.0)
                    m4 = torch.cat([full, target_rho, target_rho, target_rho], dim=1)
                elif weight_mode == "full":
                    m4 = torch.cat([full, full, full, full], dim=1)
                else:
                    m4 = torch.cat([full, dm, dm, dm], dim=1)

                if use_nlll and log_var is not None:
                    if loss_style == "teacher":
                        step_recon, sigma_mean = _teacher_style_recon(
                            x_hat, x_gt, log_var=log_var,
                            lambda_vel=teacher_lambda_vel,
                            lambda_rho=teacher_lambda_rho,
                            lambda_vx=teacher_lambda_vx,
                            lambda_vy=teacher_lambda_vy,
                        )
                    elif loss_style == "detached_nll":
                        step_recon_mse = masked_mean_sq(err * ch_w, m4)
                        err_det = (x_hat - x_gt).pow(2)[:, :3]  # already detached in val
                        nll_aux = 0.5 * torch.exp(-log_var) * err_det + 0.5 * log_var
                        step_recon = step_recon_mse + 0.1 * nll_aux.mean()
                        sigma_mean = torch.exp(0.5 * log_var).mean().detach()
                    else:
                        step_recon, sigma_mean = _heteroscedastic_nll(
                            x_hat, log_var, x_gt, ch_w, m4
                        )
                else:
                    if loss_style == "teacher":
                        step_recon, sigma_mean = _teacher_style_recon(
                            x_hat, x_gt, log_var=None, lambda_vel=teacher_lambda_vel,
                            lambda_rho=teacher_lambda_rho,
                            lambda_vx=teacher_lambda_vx,
                            lambda_vy=teacher_lambda_vy,
                        )
                    else:
                        step_recon = masked_mean_sq(err * ch_w, m4)
                        sigma_mean = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
                step_bg = _bg_suppress_term(x_hat, dm) if lambda_bg > 0 else torch.zeros((), device=device)
                step_dir = _direction_loss(x_hat, x_gt, dm) if lambda_dir > 0 else torch.zeros((), device=device)
                step_mass = _mass_conservation_loss(x_hat, x_gt) if lambda_mass > 0 else torch.zeros((), device=device)
                step_log_rho = _log_density_mse(x_hat, x_gt) if lambda_log_rho > 0 else torch.zeros((), device=device)
                step_loss = (
                    step_recon
                    + lambda_bg * step_bg
                    + lambda_dir * step_dir
                    + lambda_mass * step_mass
                    + lambda_log_rho * step_log_rho
                )

                loss_sum += float(step_loss.item())
                recon_sum += float(step_recon.item())
                bg_sum += float(step_bg.item())
                dir_sum += float(step_dir.item())
                mass_sum += float(step_mass.item())
                log_rho_sum += float(step_log_rho.item())
                if use_nlll:
                    sigma_sum += float(sigma_mean.item())
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
        "recon": recon_sum / max(steps, 1),
        "bg_suppress": bg_sum / max(steps, 1),
        "dir": dir_sum / max(steps, 1),
        "mass": mass_sum / max(steps, 1),
        "log_rho": log_rho_sum / max(steps, 1),
        "sigma_mean": sigma_sum / max(steps, 1) if use_nlll else 0.0,
    }
    for i, name in enumerate(_CH_NAMES):
        out[name] = float(ch_mse[i])
        out[f"{name}_w"] = float(ch_w_flat[i].item() * ch_mse[i] / 4.0)
    n_bg_safe = rho_bg_n.clamp_min(1.0)
    out["rho_bg_mse"] = float((rho_bg_se / n_bg_safe).item())
    out["rho_bg_mean"] = float((rho_bg_abs / n_bg_safe).item())
    return out
