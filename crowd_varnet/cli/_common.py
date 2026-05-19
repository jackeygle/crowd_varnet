"""CLI 共享辅助：从 ckpt + training_meta 构造 CrowdVarNet 并加载权重。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..datasets import CrowdVarNetDataset
from ..models import CrowdVarNet, load_frozen_pedpred


def wrap_loader_varnet(
    loader: DataLoader,
    seed: int,
    *,
    obs_mode: str = "sensor",
    partial_frac: float = 0.35,
    sensing_range: float = 5.0,
    num_agents: int = 3,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """Re-wrap an ATC base ``DataLoader`` so that each item yields the tuple
    ``(history, obs, obs_mask, x_gt)`` expected by ``CrowdVarNet`` for single-step
    inference / evaluation. Used by ``infer_cli`` and ``scripts/diag_density_leak.py``.
    """
    base_ds = loader.dataset
    ds_kw = dict(
        obs_mode=obs_mode,
        partial_frac=partial_frac,
        sensing_range=sensing_range,
        num_agents=num_agents,
    )
    if isinstance(base_ds, ConcatDataset):
        wrapped = ConcatDataset(
            [CrowdVarNetDataset(ds, seed=seed + i, **ds_kw) for i, ds in enumerate(base_ds.datasets)]
        )
    else:
        wrapped = CrowdVarNetDataset(base_ds, seed=seed, **ds_kw)

    nw = num_workers if num_workers is not None else loader.num_workers
    dl_kw: dict = dict(
        dataset=wrapped,
        batch_size=batch_size if batch_size is not None else loader.batch_size,
        shuffle=shuffle if shuffle is not None else getattr(loader, "shuffle", True),
        num_workers=nw,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=drop_last if drop_last is not None else getattr(loader, "drop_last", False),
        generator=getattr(loader, "generator", None),
    )
    if nw > 0:
        dl_kw["prefetch_factor"] = getattr(loader, "prefetch_factor", None) or 2
        dl_kw["persistent_workers"] = getattr(loader, "persistent_workers", False)
    return DataLoader(**dl_kw)


def load_training_meta(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """读 ``training_meta.json``；文件不存在返回空 dict。"""
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
    根据 ``best.pt``（同目录 ``training_meta.json``）构造并加载 ``CrowdVarNet``。

    - 自动从 state_dict 推断 ``use_gru``（是否含 ``solver.gru_cell.*`` 键）；
    - PedPred 路径从 meta 读；其他超参（``nin``、``n_iter``、``w_prior``、``rho_mask_thr``、``arch``、
      ``gru_ch``）支持通过 ``meta_overrides`` 覆盖。

    返回 ``(model.eval(), meta)``。
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
    arch = str(meta.get("arch", "pedpred3"))
    gru_ch = int(meta.get("gru_ch", 16))

    payload = torch.load(ckpt_path, map_location=device)
    sd = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    # 自动从 state_dict 推断 solver_type / init_gate（向后兼容旧 ckpt）
    keys = list(sd.keys())
    if any(str(k).startswith("solver.convgru") or str(k).startswith("solver._convgru_list") for k in keys):
        solver_type = "convgru"
    elif any(str(k).startswith("solver.gru_cell") for k in keys):
        solver_type = "gru"
    else:
        solver_type = "scalar"
    init_gate = any(str(k).startswith("init_gate.") for k in keys)
    # 推断 share_across_iter / hidden（仅 convgru 有意义）
    share_across_iter = not any(str(k).startswith("solver._convgru_list") for k in keys)
    solver_hidden = int(meta.get("solver_hidden", 32))
    solver_kernel = int(meta.get("solver_kernel", 3))
    if solver_type == "convgru":
        # 优先从 ckpt 中真实形状推断 hidden
        for k in keys:
            if k.endswith("solver.convgru.conv_rz.weight") or k.endswith("conv_rz.weight"):
                solver_hidden = int(sd[k].shape[0] // 2)
                break
    init_gate_mid = int(meta.get("init_gate_mid", 16))
    solver_dropout = float(meta.get("solver_dropout", 0.0))
    unfreeze_phi_tail = int(meta.get("unfreeze_phi_tail", 0))

    use_gru = solver_type == "gru"  # 兼容旧构造签名

    ped = load_frozen_pedpred(ped_path, device, arch=arch)
    model = CrowdVarNet(
        ped_pred=ped,
        freeze_phi=True,
        T_hist=T_hist,
        n_iter=n_iter,
        use_gru=use_gru,
        gru_ch=gru_ch,
        w_prior=w_prior,
        rho_mask_thr=rho_mask_thr,
        solver_type=solver_type,
        solver_hidden=solver_hidden,
        solver_kernel=solver_kernel,
        solver_share=share_across_iter,
        solver_dropout=solver_dropout,
        init_gate=init_gate,
        init_gate_mid=init_gate_mid,
        unfreeze_phi_tail=unfreeze_phi_tail,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, meta
