"""
CrowdVarNet 与老师 pipeline 的对比评估（同一 test 轨迹 + ``obs_replay.pkl``）。

Teacher 侧跑的是 **EnKF 同化**（老师自己的方法）；本 CLI 只跑 **CrowdVarNet 推理**。
两边 **对齐的是实验配置**：同一 replay 里的观测几何 ``C_joint``/``y_*``、同一数据 period、
与 ``get_atc_data('test')`` / ``PEDPRED_NIN=1`` 等设定；**不是**把 EnKF 算法并入 CrowdVarNet。

具体评估步骤：
- 每步使用 replay 中的 ``C_joint`` / ``y_clean`` / ``y_obs`` 几何；
- 用 **过去真值帧** 拼成长度 ``T_hist`` 的 history（与训练时「历史来自序列真值」一致）。

输出：``step_*.png``、可选 ``step_npz/``（供 ``python -m crowd_varnet.deps.merge_compare`` 与 Teacher 拼图）、
``rmse_per_step.npy``、``compare_summary.json``。

用法::

    export PYTHONPATH=/scratch/work/zhangx29/crowd_varnet
    export PEDPRED_BATCH=64 PEDPRED_NIN=1 PEDPRED_NOUT=1 \\
           PEDPRED_RESOLUTION=1.0 PEDPRED_PERIOD=1.0 PEDPRED_KERNEL=tri

    python -m crowd_varnet.compare_teacher_cli \\
      --cvn-ckpt /path/to/best.pt \\
      --obs-replay-pkl /path/to/obs_replay.pkl \\
      --run-dir /path/to/out/crowd_varnet \\
      --num-steps 300 \\
      --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from .deps.dataset_atc import get_atc_data
from .deps.enkf_sensors import GeneratePartialObs
from .deps.observation_replay import _as_eval_tensor, load_replay_bundle
from .deps.utils_plot import dump_step_arrays_npz, save_step_plot

from .core import CrowdVarNet, load_frozen_pedpred  # noqa: E402


def _load_meta(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "training_meta.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _history_tensor(
    frames_so_far: List[torch.Tensor],
    T_hist: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """frames_so_far: 每步真值 [4,H,W]；左端用最早一帧 padding 至长度 T_hist → [1,T_hist,4,H,W]。"""
    if not frames_so_far:
        raise ValueError("empty history")
    seq = frames_so_far[-T_hist:]
    if len(seq) < T_hist:
        pad = [seq[0]] * (T_hist - len(seq))
        seq = pad + seq
    return torch.stack(seq, dim=0).unsqueeze(0).to(device=device, dtype=dtype)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrowdVarNet vs teacher-aligned replay comparison")
    p.add_argument(
        "--cvn-ckpt",
        type=str,
        required=True,
        help="CrowdVarNet best.pt（同目录可有 training_meta.json）",
    )
    p.add_argument("--obs-replay-pkl", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--num-steps", type=int, default=None, help="默认：replay 全长")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--data-period", type=float, default=1.0)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--dump-step-npz", action="store_true", help="写入 step_npz/ 供 merge 脚本拼图")
    p.add_argument(
        "--T-hist",
        type=int,
        default=None,
        help="覆盖 history 长度（默认 training_meta.nin 或 5）",
    )
    args, rest = p.parse_known_args(argv if argv is not None else sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    nin_test = int(os.environ.get("PEDPRED_NIN", "1"))
    batch_test = int(os.environ.get("PEDPRED_BATCH", "64"))
    print(
        f"[data] period={float(args.data_period)} nin={nin_test} batch={batch_test}",
        flush=True,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    cvn_path = Path(args.cvn_ckpt).resolve()
    meta = _load_meta(cvn_path.parent)
    T_hist = int(args.T_hist if args.T_hist is not None else meta.get("nin", 5))
    n_iter = int(meta.get("n_iter", 8))
    w_prior = float(meta.get("w_prior", 0.5))
    rho_mask_thr = float(meta.get("rho_mask_thr", 0.05))
    arch = str(meta.get("arch", "pedpred3"))
    ped_path = meta.get("pedpred_ckpt")
    if not ped_path:
        raise SystemExit("training_meta.json 缺少 pedpred_ckpt")
    ped_path = str(Path(ped_path).resolve())

    replay_bundle = load_replay_bundle(os.path.abspath(args.obs_replay_pkl))
    entries = replay_bundle["entries"]
    GRID_SIZE = tuple(replay_bundle["GRID_SIZE"])
    STATE_SHAPE = tuple(replay_bundle["STATE_SHAPE"])
    num_steps = int(args.num_steps) if args.num_steps is not None else len(entries)
    if len(entries) < num_steps:
        raise ValueError(f"replay has {len(entries)} steps, need {num_steps}")

    params_dump = {
        "cvn_ckpt": str(cvn_path),
        "obs_replay_pkl": os.path.abspath(args.obs_replay_pkl),
        "num_steps": num_steps,
        "T_hist": T_hist,
        "data_period": args.data_period,
        "device": str(device),
        "meta": meta,
    }
    with open(os.path.join(run_dir, "compare_parameters.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(params_dump, f)

    ped = load_frozen_pedpred(ped_path, device, arch=arch)
    model = CrowdVarNet(
        ped_pred=ped,
        freeze_phi=True,
        T_hist=T_hist,
        n_iter=n_iter,
        use_gru=False,
        w_prior=w_prior,
        rho_mask_thr=rho_mask_thr,
    ).to(device)
    payload = torch.load(cvn_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    gen = GeneratePartialObs(
        GRID_SIZE,
        STATE_SHAPE,
        agent_pos=(0, 0),
        sensing_range=int(replay_bundle["sensing_range"]),
    )

    data = get_atc_data(
        "test",
        period=float(args.data_period),
        nin=nin_test,
        nout=1,
        batch=batch_test,
        num_workers=0,
        drop_last=False,
    )
    rmse_list: List[float] = []
    mae_list: List[float] = []
    rho_lst: List[float] = []
    vel_lst: List[float] = []
    var_lst: List[float] = []
    times: List[float] = []

    hist_frames: List[torch.Tensor] = []
    step_idx = 0

    for x_batch, _ in data:
        x_batch = _as_eval_tensor(x_batch)
        if x_batch.dim() != 5:
            raise ValueError(f"expected (B,T_in,C,H,W), got {tuple(x_batch.shape)}")
        if x_batch.shape[1] != 1:
            raise ValueError("need cfg nin=1; export with PEDPRED_NIN=1")

        for bi in range(x_batch.shape[0]):
            if step_idx >= num_steps:
                break

            t0 = time.time()
            true_t = x_batch[bi, 0]
            hist_frames.append(true_t.detach().clone())
            true_state = true_t.detach().cpu().numpy().astype(np.float64).flatten()
            true_hw = true_state.reshape(STATE_SHAPE)

            rec = entries[step_idx]
            C_joint = rec["C_joint"]
            y_clean = np.asarray(rec["y_clean"], dtype=np.float64)

            if step_idx == 0:
                slack = float(np.max(np.abs(C_joint @ true_state - y_clean)))
                if slack > 1e-3:
                    raise ValueError(
                        f"replay y_clean inconsistent with trajectory (max|.|={slack}); "
                        "check PEDPRED_PERIOD / batch order vs replay export."
                    )

            partial_obs = gen.reconstruct_observation(y_clean, C_joint, fill_value=np.nan)
            obs_mask_np = (~np.isnan(partial_obs[0:1])).astype(np.float32)
            obs_np = true_hw * obs_mask_np

            history = _history_tensor(hist_frames, T_hist, device, true_t.dtype)
            obs = torch.from_numpy(obs_np).to(device=device, dtype=history.dtype).unsqueeze(0)
            obs_mask = torch.from_numpy(obs_mask_np).to(device=device, dtype=history.dtype).unsqueeze(0)
            x_gt = true_t.unsqueeze(0).to(device=device, dtype=history.dtype)

            x_hat = model.forward(history, obs, obs_mask, x_gt)
            est = x_hat[0].detach().cpu().numpy().astype(np.float64).reshape(-1)

            spread = np.zeros(STATE_SHAPE, dtype=np.float64)
            spread[0] = np.abs(est.reshape(STATE_SHAPE)[0] - true_hw[0])

            if not args.no_plot:
                save_step_plot(partial_obs, true_state, est, spread, step_idx, run_dir)
            if args.dump_step_npz:
                dump_step_arrays_npz(
                    run_dir,
                    step_idx,
                    partial_obs,
                    true_state,
                    est,
                    spread,
                    state_shape=STATE_SHAPE,
                )

            diff = est - true_state
            rmse = float(np.sqrt(np.mean(diff**2)))
            mae = float(np.mean(np.abs(diff)))
            er = est.reshape(STATE_SHAPE)
            rho = float(np.sqrt(np.mean((er[0] - true_hw[0]) ** 2)))
            vel = float(np.sqrt(np.mean((er[1:3] - true_hw[1:3]) ** 2)))
            vr = float(np.sqrt(np.mean((er[3] - true_hw[3]) ** 2)))

            rmse_list.append(rmse)
            mae_list.append(mae)
            rho_lst.append(rho)
            vel_lst.append(vel)
            var_lst.append(vr)
            times.append(time.time() - t0)

            print(
                f"  Step {step_idx:3d} | RMSE={rmse:.4f} | MAE={mae:.4f} | "
                f"rho_RMSE={rho:.4f} | vel_RMSE={vel:.4f} | var_RMSE={vr:.4f} | "
                f"time={times[-1]:.3f}s",
                flush=True,
            )
            step_idx += 1

        if step_idx >= num_steps:
            break

    r = np.asarray(rmse_list)
    np.save(os.path.join(run_dir, "rmse_per_step.npy"), r)

    summary = {
        "mean_rmse": float(r.mean()),
        "std_rmse": float(r.std()),
        "mean_mae": float(np.mean(mae_list)),
        "mean_rho_rmse": float(np.mean(rho_lst)),
        "mean_vel_rmse": float(np.mean(vel_lst)),
        "mean_var_rmse": float(np.mean(var_lst)),
        "mean_step_time_s": float(np.mean(times)),
        "num_steps": int(len(rmse_list)),
        "T_hist": T_hist,
    }
    with open(os.path.join(run_dir, "compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean RMSE: {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}", flush=True)
    print(
        f"Channel RMSE | rho={summary['mean_rho_rmse']:.4f} "
        f"vel={summary['mean_vel_rmse']:.4f} var={summary['mean_var_rmse']:.4f}",
        flush=True,
    )
    print(f"Results → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
