"""CLI 共享辅助：从 ckpt + training_meta 构造 CrowdVarNet 并加载权重。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ..models import CrowdVarNet, load_frozen_pedpred


def load_training_meta(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """读 training_meta.json；文件不存在返回空 dict。"""
    p = Path(run_dir) / "training_meta.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def build_model_from_ckpt(
    ckpt_path: Union[str, Path],
    *,
    device: torch.device,
    meta_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[CrowdVarNet, Dict[str, Any]]:
    """
    根据 best.pt（同目录 training_meta.json）构造并加载 CrowdVarNet。

    自动从 state_dict 推断 solver 配置（hidden、share_across_iter 等）。
    返回 (model.eval(), meta)。
    """
    ckpt_path = Path(ckpt_path).resolve()
    meta = load_training_meta(ckpt_path.parent)
    if meta_overrides:
        meta = {**meta, **{k: v for k, v in meta_overrides.items() if v is not None}}

    ped_path = meta.get("pedpred_ckpt")
    if not ped_path:
        raise SystemExit(f"{ckpt_path.parent}/training_meta.json 缺少 pedpred_ckpt")
    ped_path = str(Path(ped_path).resolve())

    T_hist = int(meta.get("nin", 5))
    n_iter = int(meta.get("n_iter", 8))
    w_prior = float(meta.get("w_prior", 0.5))
    rho_mask_thr = float(meta.get("rho_mask_thr", 0.05))
    arch = str(meta.get("arch", "pedpred3_gru_mid"))
    solver_dropout = float(meta.get("solver_dropout", 0.0))
    unfreeze_phi_tail = int(meta.get("unfreeze_phi_tail", 0))

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    # Handle key rename: old checkpoints use "solver.convgru.*", new code uses "solver.rnn_cell.*"
    remapped_sd = {}
    for k, v in sd.items():
        new_k = k.replace("solver.convgru.", "solver.rnn_cell.")
        remapped_sd[new_k] = v
    sd = remapped_sd

    # 从 state_dict 推断 solver 配置
    keys = list(sd.keys())
    share_across_iter = not any(
        str(k).startswith("solver._convgru_list") or str(k).startswith("solver._rnn_list")
        for k in keys
    )
    solver_hidden = int(meta.get("solver_hidden", 256))
    solver_kernel = int(meta.get("solver_kernel", 3))
    # 优先从 ckpt 中真实形状推断 hidden
    for k in keys:
        if ("conv_rz.weight" in k or "conv_gates.weight" in k) and "solver" in k:
            if "conv_gates" in k:
                solver_hidden = int(sd[k].shape[0] // 4)  # LSTM: 4*hidden
            else:
                solver_hidden = int(sd[k].shape[0] // 2)  # GRU: 2*hidden
            break

    ch_weights_raw = meta.get("ch_weights", [1.0, 1.0, 1.0, 0.0])
    ch_weights = tuple(float(x) for x in ch_weights_raw)
    predict_uncertainty = bool(meta.get("predict_uncertainty", False))

    # Detect rnn_type from state_dict keys
    has_lstm_keys = any("conv_gates" in k for k in keys)
    rnn_type = "lstm" if has_lstm_keys else "gru"

    # Detect attention
    has_attention = any("attention" in k for k in keys)

    # Detect momentum
    has_momentum = any("momentum_coeff" in k for k in keys)
    momentum_beta = 0.5 if has_momentum else 0.0

    # Detect obs_encoder (Perceiver-style)
    has_obs_encoder = any("obs_encoder" in k or "cross_attn" in k for k in keys)

    ped = load_frozen_pedpred(ped_path, device, arch=arch)
    model = CrowdVarNet(
        ped_pred=ped,
        freeze_phi=True,
        T_hist=T_hist,
        n_iter=n_iter,
        w_prior=w_prior,
        ch_weights=ch_weights,
        rho_mask_thr=rho_mask_thr,
        solver_hidden=solver_hidden,
        solver_kernel=solver_kernel,
        solver_share=share_across_iter,
        solver_dropout=solver_dropout,
        unfreeze_phi_tail=unfreeze_phi_tail,
        predict_uncertainty=predict_uncertainty,
        solver_use_attention=has_attention,
        solver_attn_heads=4,
        solver_momentum=momentum_beta,
        solver_rnn_type=rnn_type,
        solver_use_obs_encoder=has_obs_encoder,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, meta
