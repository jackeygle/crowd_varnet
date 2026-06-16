"""冻结 PedPred 先验：加载 ckpt + 包一层 nn.Module。"""
from __future__ import annotations

import warnings
from typing import Any, Union

import torch
import torch.nn as nn

from ..deps.grid_data import GridData
from ..deps import pedpred_models as ped_models


def load_frozen_pedpred(
    path: str,
    device: Union[torch.device, str],
    arch: str = "pedpred3_gru_mid",
    ckpt_key: str = "model",
) -> nn.Module:
    """Load a trained PedPred3_gru_mid teacher (only supported arch)."""

    arch_l = arch.lower()
    if arch_l != "pedpred3_gru_mid":
        raise ValueError(
            f"Unknown arch={arch!r}; only 'pedpred3_gru_mid' is supported"
        )
    m = ped_models.PedPred3_gru_mid()

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA 不可用，已在 CPU 上加载 PedPred。", UserWarning, stacklevel=2)
        dev = torch.device("cpu")
    if str(path).endswith(".hkl"):
        import hickle
        ckpt = hickle.load(path)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt[ckpt_key] if ckpt_key in ckpt else ckpt
    m.load_state_dict(state, strict=True)
    m.to(dev)
    m.eval()
    return m


class FrozenPedPredPrior(nn.Module):
    """冻结的 PedPred 教师：history ``[B,T,4,H,W]`` → 先验场 ``x_prior`` ``[B,4,H,W]``。

    可选解冻 ``forecaster`` 末端最后 N 层（默认 0=完全冻结，向后兼容）。
    解冻后 forward 不再 ``@torch.no_grad``，允许梯度反传到教师。
    """

    def __init__(
        self,
        ped_pred: nn.Module,
        freeze: bool = True,
        unfreeze_tail_layers: int = 0,
    ):
        super().__init__()
        self.phi = ped_pred
        # 默认全部冻结
        if freeze:
            for p in self.phi.parameters():
                p.requires_grad_(False)
            self.phi.eval()
        # 选择性解冻 forecaster 末端：只挑那些 .num 在最后 N 个 module 的 conv 层
        # PedPred_v4 的 forecaster 是 SequentialRNN，末端有 forecaster.7 / .9 / .11 是普通 Conv2d
        # 我们按层号倒序解冻（11→9→7→...）
        self.unfreeze_tail_layers = int(unfreeze_tail_layers)
        if self.unfreeze_tail_layers > 0:
            self._unfreeze_tail()

    def _unfreeze_tail(self):
        """按 named_parameters 倒序找前 N 个最后的非 STLSTM/STLSTM-like 卷积层并解冻。"""
        forecaster = getattr(self.phi, "forecaster", None)
        if forecaster is None:
            return
        # 收集 forecaster 中可见的层（按数值索引降序），筛选 Conv2d 类（非递归 STLSTM）
        from torch.nn import Conv2d, ConvTranspose2d
        candidate_layers = []
        for name, m in forecaster.named_children():
            if isinstance(m, (Conv2d, ConvTranspose2d)):
                try:
                    idx = int(name)
                except ValueError:
                    continue
                candidate_layers.append((idx, name, m))
        candidate_layers.sort(reverse=True)  # 末端优先
        unfrozen_names = []
        for idx, name, m in candidate_layers[: self.unfreeze_tail_layers]:
            for p in m.parameters():
                p.requires_grad_(True)
            # 解冻的层切回 train mode（不影响 STLSTM 等递归层）
            m.train()
            unfrozen_names.append(f"forecaster.{name}")
        if unfrozen_names:
            n_params = sum(p.numel() for n, p in self.phi.named_parameters() if p.requires_grad)
            print(
                f"[FrozenPedPredPrior] unfrozen tail layers: {unfrozen_names}  "
                f"trainable_params_in_phi={n_params}",
                flush=True,
            )

    @staticmethod
    def _unwrap_prediction(output: Any) -> torch.Tensor:
        if isinstance(output, GridData) or hasattr(output, "as_tensor"):
            t = output.as_tensor("density", "vel_mean", "vel_var")
        else:
            t = output if torch.is_tensor(output) else torch.as_tensor(output)

        # Force pure torch.Tensor (GridData is a tensor subclass that breaks arithmetic)
        if type(t) is not torch.Tensor:
            t = torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)

        if t.dim() == 5:
            t = t[:, 0]
        elif t.dim() != 4:
            raise ValueError(f"Expected [B,T,4,H,W] or [B,4,H,W], got shape {tuple(t.shape)}")
        c = t.shape[-3]
        return torch.cat([t[..., i : i + 1, :, :] for i in range(c)], dim=-3).contiguous()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: [B,T,4,H,W] → x_prior: [B,4,H,W]，与 history 同设备。"""
        dev = history.device
        # Ensure pure tensor input for GridData wrapping
        if type(history) is not torch.Tensor:
            history = torch.empty(history.shape, dtype=history.dtype, device=history.device).copy_(history)
        inp = GridData(history)
        # 全部冻结时禁用 grad，省内存；任意解冻时保持 grad 流。
        any_trainable = any(p.requires_grad for p in self.phi.parameters())
        if any_trainable:
            out = self.phi(inp, horizon=1)
        else:
            with torch.no_grad():
                out = self.phi(inp, horizon=1)
        result = self._unwrap_prediction(out)
        # Guarantee pure torch.Tensor (not a GridData subclass)
        if type(result) is not torch.Tensor:
            result = torch.empty(result.shape, dtype=result.dtype, device=result.device).copy_(result)
        return result.to(dev)
