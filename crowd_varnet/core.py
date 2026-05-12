"""
CrowdVarNet — PedPred 背景场 + 学习梯度下降的变分状态重建（4DVarNet 风格）。

``GridData`` 与 PedPred 定义在包内 ``crowd_varnet.deps``。
"""
from __future__ import annotations

import argparse
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .deps.grid_data import GridData
from .deps import pedpred_models as ped_models
from .deps.enkf_sensors import MultiAgent


def load_frozen_pedpred(
    path: str,
    device: Union[torch.device, str],
    arch: str = "pedpred3",
    ckpt_key: str = "model",
) -> nn.Module:
    """加载已训练 PedPred / PedPred2 / PedPred3。"""

    arch_l = arch.lower()
    if arch_l == "pedpred3":
        m = ped_models.PedPred3()
    elif arch_l == "pedpred3_partial_observation":
        try:
            from partial_observation_experiments.models import PedPred3 as _PedPred3
        except ImportError as e:
            raise ImportError(
                "arch=pedpred3_partial_observation 需在 PYTHONPATH 中包含 project_analysis。"
            ) from e
        m = _PedPred3()
    elif arch_l == "pedpred2":
        try:
            from partial_observation_experiments.models import PedPred2 as _PedPred2
        except ImportError as e:
            raise ImportError(
                "arch=pedpred2 需在 PYTHONPATH 中包含 project_analysis。"
            ) from e
        m = _PedPred2()
    elif arch_l == "pedpred":
        try:
            from partial_observation_experiments.models import PedPred as _PedPred
        except ImportError as e:
            raise ImportError(
                "arch=pedpred 需在 PYTHONPATH 中包含 project_analysis。"
            ) from e
        m = _PedPred()
    else:
        raise ValueError(
            f"Unknown arch={arch!r}; use pedpred | pedpred2 | pedpred3 | pedpred3_partial_observation"
        )

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA 不可用，已在 CPU 上加载 PedPred。", UserWarning, stacklevel=2)
        dev = torch.device("cpu")
    # 先加载到 CPU，避免在无 GPU 节点上用 map_location=cuda 反序列化失败
    if str(path).endswith('.hkl'):
        import hickle
        ckpt = hickle.load(path)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt[ckpt_key] if ckpt_key in ckpt else ckpt
    m.load_state_dict(state, strict=True)
    m.to(dev)
    m.eval()
    return m


class PedPredAdapter(nn.Module):
    """history [B,T,4,H,W] → x_prior [B,4,H,W]（线性空间）。"""

    def __init__(
        self,
        ped_pred: nn.Module,
        freeze: bool = True,
    ):
        super().__init__()
        self.phi = ped_pred
        if freeze:
            for p in self.phi.parameters():
                p.requires_grad_(False)
            self.phi.eval()

    @staticmethod
    def _unwrap_prediction(output: Any) -> torch.Tensor:
        if isinstance(output, GridData) or hasattr(output, "as_tensor"):
            t = output.as_tensor("density", "vel_mean", "vel_var")
        else:
            t = output if torch.is_tensor(output) else torch.as_tensor(output)

        if t.dim() == 5:
            t = t[:, 0]
        elif t.dim() != 4:
            raise ValueError(f"Expected [B,T,4,H,W] or [B,4,H,W], got shape {tuple(t.shape)}")
        # 避免部分 GridData 子类在 ``pred - tgt`` 等运算上与另一子类不兼容
        c = t.shape[-3]
        return torch.cat([t[..., i : i + 1, :, :] for i in range(c)], dim=-3).contiguous()

    @torch.no_grad()
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: [B,T,4,H,W] → x_prior: [B,4,H,W]，与 history 同设备。"""
        dev = history.device
        inp = GridData(history)
        out = self.phi(inp, horizon=1)
        return self._unwrap_prediction(out).to(dev)


class VariationalCost(nn.Module):
    def __init__(
        self,
        w_obs: float = 1.0,
        w_prior: float = 0.5,
        ch_weights: Tuple[float, float, float, float] = (2.5, 1.5, 1.0, 0.5),
        rho_mask_thr: float = 0.05,
        prior_use_ch_weights: bool = True,
    ):
        super().__init__()
        self.w_obs = w_obs
        self.w_prior = w_prior
        self.rho_mask_thr = rho_mask_thr
        self.prior_use_ch_weights = prior_use_ch_weights
        w = torch.tensor(ch_weights, dtype=torch.float32).view(1, 4, 1, 1)
        self.register_buffer("ch_w", w)

    def obs_term(self, x: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        err = (x - obs).pow(2)
        rho_mask = (obs[:, 0:1] > self.rho_mask_thr).float()
        # 与 vel/var 一致：无观测密度处不计入（obs 在未观测格为 0，rho_mask 自然为 0）
        combined = torch.cat(
            [
                obs_mask * rho_mask,
                obs_mask * rho_mask,
                obs_mask * rho_mask,
                obs_mask * rho_mask,
            ],
            dim=1,
        )
        return (err * self.ch_w * combined).sum() / (combined.sum() + 1e-6)

    def prior_term(
        self, x: torch.Tensor, x_prior: torch.Tensor, density_mask: torch.Tensor
    ) -> torch.Tensor:
        err = (x - x_prior).pow(2)
        if self.prior_use_ch_weights:
            err = err * self.ch_w
        m4 = density_mask.expand_as(x)
        return (err * m4).sum() / (m4.sum() + 1e-6)

    def forward(
        self,
        x: torch.Tensor,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        x_prior: torch.Tensor,
        density_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o = self.obs_term(x, obs, obs_mask)
        p = self.prior_term(x, x_prior, density_mask)
        return self.w_obs * o + self.w_prior * p, o, p


def clip_crowd_state(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[:, 0:1].clamp_(0.0, 5.0)
    out[:, 1:3].clamp_(-5.0, 5.0)
    out[:, 3:4].clamp_(0.0, 2.0)
    return out


def density_support_mask(x_gt: torch.Tensor, rho_thr: float) -> torch.Tensor:
    """Binary mask [B,1,H,W]: 1 where ground-truth density exceeds ``rho_thr``."""
    return (x_gt[:, 0:1] > rho_thr).to(dtype=x_gt.dtype)


def masked_mean_sq(err: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean of ``err`` over entries where ``mask`` is 1; ``mask`` broadcastable to ``err``."""
    m = mask
    if m.shape != err.shape:
        m = m.expand_as(err)
    den = m.sum().clamp_min(eps)
    return (err * m).sum() / den


class LearnedGradSolver(nn.Module):
    def __init__(
        self,
        n_iter: int = 8,
        use_gru: bool = False,
        gru_ch: int = 16,
        clip_each_step: bool = True,
    ):
        super().__init__()
        self.n_iter = n_iter
        self.use_gru = use_gru
        self.clip_each_step = clip_each_step
        self.gru_ch = gru_ch

        if not use_gru:
            self.log_steps = nn.Parameter(torch.full((n_iter,), -2.0))
        else:
            self.gru_cell = nn.GRUCell(input_size=8, hidden_size=gru_ch)
            self.gru_out = nn.Sequential(
                nn.Linear(gru_ch, 4),
                nn.Tanh(),
            )

    def forward(
        self,
        x_init: torch.Tensor,
        cost_fn: VariationalCost,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        x_prior: torch.Tensor,
        density_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x_init.clone().detach().requires_grad_(True)
        B, C, H, W = x.shape
        h = (
            torch.zeros(B * H * W, self.gru_ch, device=x.device, dtype=x.dtype)
            if self.use_gru
            else None
        )

        for k in range(self.n_iter):
            cost, _, _ = cost_fn(x, obs, obs_mask, x_prior, density_mask)
            grad = torch.autograd.grad(cost, x, create_graph=True)[0]

            if not self.use_gru:
                x = x - torch.exp(self.log_steps[k]) * grad
            else:
                assert h is not None
                feat = torch.cat([grad, x.detach()], dim=1)
                feat_flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C * 2)
                h = self.gru_cell(feat_flat, h)
                delta = self.gru_out(h).reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = x - delta

            if self.clip_each_step:
                x = clip_crowd_state(x)
            x = x.requires_grad_(True)

        return x


class CrowdVarNet(nn.Module):
    """
    训练与验证的 ``recon``、分量 MSE、``phi_mse`` 以及求解器内的 ``prior_term`` 均在
    ``x_gt`` 密度大于 ``rho_mask_thr`` 的格点上平均；``forward`` 需提供 ``x_gt`` 以构造该掩码。
    """

    def __init__(
        self,
        ped_pred: nn.Module,
        freeze_phi: bool = True,
        T_hist: int = 5,
        n_iter: int = 8,
        use_gru: bool = False,
        gru_ch: int = 16,
        w_obs: float = 1.0,
        w_prior: float = 0.5,
        ch_weights: Tuple[float, float, float, float] = (2.5, 1.5, 1.0, 0.5),
        rho_mask_thr: float = 0.05,
        prior_use_ch_weights: bool = True,
        clip_solver_steps: bool = True,
    ):
        super().__init__()
        self.T_hist = T_hist
        self.adapter = PedPredAdapter(ped_pred, freeze=freeze_phi)
        self.cost_fn = VariationalCost(
            w_obs,
            w_prior,
            ch_weights,
            rho_mask_thr,
            prior_use_ch_weights=prior_use_ch_weights,
        )
        self.solver = LearnedGradSolver(
            n_iter,
            use_gru=use_gru,
            gru_ch=gru_ch,
            clip_each_step=clip_solver_steps,
        )

    def forward(
        self, history: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor, x_gt: torch.Tensor
    ) -> torch.Tensor:
        x_prior = self.adapter(history)
        density_mask = density_support_mask(x_gt, self.cost_fn.rho_mask_thr)
        x_init = obs * obs_mask + x_prior * (1.0 - obs_mask)
        return self.solver(x_init, self.cost_fn, obs, obs_mask, x_prior, density_mask)

    def compute_loss(
        self,
        history: torch.Tensor,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        x_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_hat = self.forward(history, obs, obs_mask, x_gt)
        dm = density_support_mask(x_gt, self.cost_fn.rho_mask_thr)
        m4 = dm.expand_as(x_gt)
        recon_loss = masked_mean_sq(
            (x_hat - x_gt).pow(2) * self.cost_fn.ch_w, m4
        )

        with torch.no_grad():
            x_prior = self.adapter(history)
            _, obs_t, prior_t = self.cost_fn(x_hat, obs, obs_mask, x_prior, dm)
            phi_mse = masked_mean_sq((x_prior - x_gt).pow(2), m4)

        rho_m = dm  # [B,1,H,W]
        info = {
            "recon": float(recon_loss.item()),
            "obs_term": float(obs_t.item()),
            "prior_term": float(prior_t.item()),
            "phi_mse": float(phi_mse.item()),
            "rho_mse": float(
                masked_mean_sq((x_hat[:, 0:1] - x_gt[:, 0:1]).pow(2), rho_m).item()
            ),
            "vx_mse": float(
                masked_mean_sq((x_hat[:, 1:2] - x_gt[:, 1:2]).pow(2), rho_m).item()
            ),
            "vy_mse": float(
                masked_mean_sq((x_hat[:, 2:3] - x_gt[:, 2:3]).pow(2), rho_m).item()
            ),
            "var_mse": float(
                masked_mean_sq((x_hat[:, 3:4] - x_gt[:, 3:4]).pow(2), rho_m).item()
            ),
            "density_mask_frac": float(dm.mean().item()),
        }
        return recon_loss, info


def spatial_sensor_mask(
    H: int,
    W: int,
    agents_rc: Sequence[Tuple[int, int]],
    sensing_range: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    部分观测的**几何**与 PA/ENKF 脚本里 ``GeneratePartialObs.cells_within_range`` 相同（实验配置一致）：
    格点 (r,c) 满足 ``sqrt((r-r0)^2 + (c-c0)^2) <= sensing_range`` 视为可见；多智能体圆盘并集。
    注意：这里只是**怎么画可见区域**，不涉及 EnKF 同化或滤波。返回 ``obs_mask`` 形状 ``[1, H, W]``。
    """
    rr = torch.arange(H, dtype=dtype, device=device).view(H, 1)
    cc = torch.arange(W, dtype=dtype, device=device).view(1, W)
    mask = torch.zeros(H, W, dtype=dtype, device=device)
    sr = float(sensing_range)
    sr2 = sr * sr
    for r0, c0 in agents_rc:
        dist_sq = (rr - float(r0)) ** 2 + (cc - float(c0)) ** 2
        mask = torch.maximum(mask, (dist_sq <= sr2).to(dtype))
    return mask.unsqueeze(0)


def _target_frame_index_in_episode(seq_ds: torch.utils.data.Dataset, local_window_idx: int) -> int:
    """H5 内目标帧下标（与 ``SeqDataset`` 的 ``start + input_len`` 一致）。"""
    if not hasattr(seq_ds, "input_len") or not hasattr(seq_ds, "step"):
        raise TypeError("CrowdVarNetDataset.seq_ds must be a SeqDataset (input_len, step).")
    step = seq_ds.step
    if callable(step):
        step = int(step())
    else:
        step = int(step)
    return int(local_window_idx * step + int(seq_ds.input_len))


def stack_grid_sequence(seq: List[Any]) -> torch.Tensor:
    tensors = []
    for f in seq:
        if isinstance(f, GridData) or hasattr(f, "as_tensor"):
            tensors.append(GridData(f).as_tensor("density", "vel_mean", "vel_var"))
        else:
            tensors.append(torch.as_tensor(f))
    return torch.stack(tensors, dim=0)


class CrowdVarNetDataset(torch.utils.data.Dataset):
    """
    从 ``SeqDataset`` 取 history / x_gt，并构造部分观测。

    **定位（避免和 baseline 搞混）**  
    CrowdVarNet 学的是「部分观测 + PedPred 先验 → 重建全场」的变分网络，**不实现、也不训练**
    EnKF/LEnKF 的同化算法；那些仍是 teacher/baseline **自己的方法**。

    这里要对齐的是 **实验配置**：与 PA 脚本里用于**生成部分观测**的那套设定一致——
    ``MultiAgent`` 的 goal/``move_agents`` 叙事、``num_agents``、``sensing_range``、圆盘并集几何——
    以便和对比实验 **同一观测条件**。不是把 EnKF 整套流程搬进本模型。

    **模式**  
    - ``obs_mode="sensor"``（默认）：按上面对齐方式，每个 episode（通常一个 H5）固定 RNG 初始化 agent，
      对目标帧下标 ``G`` 执行 ``G+1`` 次 ``move_agents()`` 再算 mask（与 PA 里「先动再采观测」的**观测侧**步序一致）。
      每样本约 ``O(G)``，靠后窗口 ``G`` 大时 epoch 会慢于 ``sensor_static``，属为保持**观测配置**一致的成本。
    - ``obs_mode="sensor_static"``：每样本独立随机圆心（不做 goal 运动，仅消融）。
    - ``obs_mode="random"``：随机格子比例 ``partial_frac``（消融）。
    """

    def __init__(
        self,
        seq_ds: torch.utils.data.Dataset,
        *,
        obs_mode: str = "sensor",
        partial_frac: float = 0.35,
        sensing_range: float = 5.0,
        num_agents: int = 3,
        seed: int = 0,
    ):
        self.seq_ds = seq_ds
        self.obs_mode = obs_mode.lower().strip()
        if self.obs_mode not in ("sensor", "sensor_static", "random"):
            raise ValueError(
                f"obs_mode must be 'sensor', 'sensor_static', or 'random', got {self.obs_mode!r}"
            )
        self.partial_frac = partial_frac
        self.sensing_range = float(sensing_range)
        self.num_agents = max(1, int(num_agents))
        self._seed = int(seed)

    def __len__(self):
        return len(self.seq_ds)

    def __getitem__(self, idx: int):
        inp, tgt = self.seq_ds[idx]
        history = stack_grid_sequence(inp)
        x_gt = stack_grid_sequence(tgt)[0]

        _, _, H, W = history.shape

        if self.obs_mode == "sensor":
            G = _target_frame_index_in_episode(self.seq_ds, idx)
            rng_agents = np.random.RandomState(int(self._seed))
            agents_ma = MultiAgent(
                (H, W),
                (4, H, W),
                sensing_range=self.sensing_range,
                num_agents=self.num_agents,
                rng=rng_agents,
            )
            for _ in range(G + 1):
                agents_ma.move_agents()
            obs_mask = spatial_sensor_mask(
                H, W, list(agents_ma.positions), float(self.sensing_range)
            )
        elif self.obs_mode == "sensor_static":
            rng = np.random.RandomState(self._seed + idx * 100003)
            agents = [
                (int(rng.randint(0, H)), int(rng.randint(0, W)))
                for _ in range(self.num_agents)
            ]
            obs_mask = spatial_sensor_mask(H, W, agents, self.sensing_range)
        else:
            flat = H * W
            n_vis = max(1, int(flat * self.partial_frac))
            g = torch.Generator()
            g.manual_seed(self._seed + idx)
            perm = torch.randperm(flat, generator=g)
            vis_idx = perm[:n_vis]
            obs_mask = torch.zeros(1, H, W)
            obs_mask.reshape(-1)[vis_idx] = 1.0

        obs = x_gt * obs_mask

        return history, obs, obs_mask, x_gt


def train_one_epoch(
    model: CrowdVarNet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    log_interval: int = 1,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    eprefix = (
        f"[{int(epoch) + 1}/{int(num_epochs)}] "
        if epoch is not None and num_epochs is not None
        else ""
    )
    n_batches = len(loader)
    for batch in loader:
        history, obs, obs_mask, x_gt = batch
        history = history.to(device)
        obs = obs.to(device)
        obs_mask = obs_mask.to(device)
        x_gt = x_gt.to(device)

        loss, info = model.compute_loss(history, obs, obs_mask, x_gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
        n += 1
        if log_interval > 0 and (n % log_interval == 0 or n == n_batches):
            print(
                f"{eprefix}batch {n}/{n_batches}  loss={loss.item():.6f}"
                f"  rho={info['rho_mse']:.4f}  vx={info['vx_mse']:.4f}  vy={info['vy_mse']:.4f}",
                flush=True,
            )
    return total / max(n, 1)


def parse_smoke_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrowdVarNet smoke")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def run_smoke() -> None:
    args = parse_smoke_args()
    device = torch.device(args.device)
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
        n_iter=6,
        use_gru=False,
    ).to(device)

    history = torch.rand(B, T, 4, H, W, device=device)
    obs_mask = torch.zeros(B, 1, H, W, device=device)
    obs_mask[:, :, :, : W // 2] = 1.0
    obs = torch.rand(B, 4, H, W, device=device) * obs_mask
    x_gt = torch.rand(B, 4, H, W, device=device)

    loss, info = model.compute_loss(history, obs, obs_mask, x_gt)
    print("loss", loss.item(), "phi_mse", info["phi_mse"], "recon", info["recon"])
