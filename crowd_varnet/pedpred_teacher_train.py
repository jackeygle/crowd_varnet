"""
PedPred teacher (ATC) training: NLLL only, pedpred3_gru_mid backbone only.

Examples::

    python -u -m crowd_varnet.pedpred_teacher_train [--max-epochs N] [RESUME_GLOB]
"""
from __future__ import annotations

from collections import defaultdict, namedtuple
import os
import sys
import time

import matplotlib as mpl
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .deps.dataset_atc import get_atc_data
from .deps.grid_data import GridData
from .deps.pedpred_metrics import Metrics
from .deps.pedpred_models import PedPred3_gru_mid
from .deps.pedpred_train_utils import CatchSignal, cuda_context
from .deps.pedpred_training_state import PedPredTeacherState
from .pedpred_teacher_config import cfg


def _maybe_flip_w(input, target, prob: float):
    """Flip (input, target) along the W axis with probability ``prob``; vx (vel_mean[0]) is negated."""
    if prob <= 0.0:
        return input, target
    if torch.rand(()).item() >= prob:
        return input, target

    def _flip(x):
        t = GridData(x).as_tensor('density', 'vel_mean', 'vel_var')
        t = torch.flip(t, dims=(-1,)).contiguous()
        t[..., 1, :, :] = -t[..., 1, :, :]   # vx 取反
        return GridData(t)

    return _flip(input), _flip(target)


loss_metric = cfg.loss


def scalar_training_loss(metrics: Metrics):
    if cfg.loss_vel_scale != 1.0 and loss_metric == "mean total weighted NLLL":
        den = metrics["mean NLLL density"]
        vel = metrics["mean weighted NLLL vel_est"] + metrics["mean weighted NLLL vel_unc"]
        return den + cfg.loss_vel_scale * vel
    return metrics[loss_metric]


def batch_loss(pred, target):
    return scalar_training_loss(Metrics(pred, target))


def batch_loss_and_decomp(pred, target):
    m = Metrics(pred, target)
    loss = scalar_training_loss(m)
    den = m["mean NLLL density"]
    vel = m["mean weighted NLLL vel_est"] + m["mean weighted NLLL vel_unc"]
    vel_x = m["mean weighted NLLL vel_est_vx"]
    vel_y = m["mean weighted NLLL vel_est_vy"]
    return loss, den, vel, vel_x, vel_y


def _fmt_nlll_postfix(loss, den, vel, vel_x, vel_y) -> str:
    """L_vel = vel_est+vel_unc (weighted mean); L_velx/L_vely split only the vel_est part."""
    L = float(loss.detach())
    d = float(den.detach())
    v = float(vel.detach())
    vx = float(vel_x.detach())
    vy = float(vel_y.detach())
    return f"L={L:.3f} den={d:.3f} vel={v:+.3f} velx={vx:+.3f} vely={vy:+.3f}"


def get_teacher_data(
    *mode: str,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    validation_num_workers: int | None = None,
):
    mode = mode or ("train", "valid")
    dataset_name, _, subset = cfg.dataset.partition(":")
    if dataset_name != "atc":
        raise ValueError(
            f"crowd_varnet.pedpred_teacher_train only supports dataset 'atc:*'; got {cfg.dataset!r}"
        )
    if not subset:
        subset = "corridor"
    val_nw = num_workers if validation_num_workers is None else validation_num_workers
    data: dict[str, DataLoader] = {}
    for m in mode:
        nw = num_workers if m == "train" else val_nw
        single = get_atc_data(
            m,
            batch=cfg.batch,
            nin=cfg.nin,
            nout=cfg.nout,
            resolution=cfg.resolution,
            period=cfg.period,
            kernel=cfg.kernel,
            subset=subset,
            num_workers=nw,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            validation_num_workers=None,
        )
        data[m] = single

    DataLoaderSet = namedtuple("DataLoaderSet", list(data.keys()))
    out = DataLoaderSet(**data)
    if len(out) == 1:
        return out[0]
    return out


class EDException(Exception):
    """Early Death (NaN / unstable early steps)."""

    def __init__(self, name=None, age=None, cause=None):
        self.name = name
        self.age = age
        self.cause = cause

    def __str__(self):
        lines = ["R I P"]
        if self.name:
            lines += [self.name]
        if self.age is not None:
            lines += [f"age {self.age}"]
        if self.cause:
            lines += [f"died with {self.cause}"]
        lines += [time.strftime("%e %B, %Y")]
        return "\n".join(f"{line:^25s}" for line in lines)


def get_trainstate(file_glob, live_cfg):
    arch = os.environ.get("PEDPRED_ARCH", "pedpred3_gru_mid").lower().strip()
    if arch != "pedpred3_gru_mid":
        raise ValueError(
            f"PEDPRED_ARCH must be 'pedpred3_gru_mid' (only supported arch), got {arch!r}"
        )
    model = PedPred3_gru_mid()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[get_trainstate] arch={arch} total_params={n_params:,}", flush=True)
    if torch.cuda.is_available():
        model = model.cuda()
    wd = float(getattr(cfg, "weight_decay", 0.0))
    if wd > 0:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
                                amsgrad=True, weight_decay=wd)
        opt_name = "AdamW"
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), amsgrad=True)
        opt_name = "Adam"
    print(f"[get_trainstate] optimizer={opt_name} lr={cfg.lr} wd={wd:.2e}", flush=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, cooldown=0, threshold=0)
    return PedPredTeacherState(
        file_glob,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        live_cfg=live_cfg,
    )


def train(state: PedPredTeacherState, data: DataLoader):
    state.model.train()
    model_device = next(state.model.parameters()).device
    losses = []
    decomp_den = []
    decomp_vel = []
    decomp_vel_x = []
    decomp_vel_y = []
    data = tqdm(data, desc=f"{state.name} training, epoch {state.epochs:,}")
    for _, (input, target) in enumerate(data):
        input, target = _maybe_flip_w(input, target, getattr(cfg, "flip_w_prob", 0.0))
        input = input.to(model_device)
        target = target.to(model_device)
        pred = state.model(input, horizon=target.shape[1])
        loss, den_t, vel_t, vx_t, vy_t = batch_loss_and_decomp(pred, target)
        data.set_postfix_str(_fmt_nlll_postfix(loss, den_t, vel_t, vx_t, vy_t))
        if not loss.isfinite():
            if state.steps < 18:
                raise EDException(name=state.name, age=state.steps, cause=f"{loss=:g}")
            print(f"!!! {loss=:g}, skipping iteration")
            continue
        loss.backward()
        clip_grad_value_(state.model.parameters(), 1e3)
        grad_norm = clip_grad_norm_(state.model.parameters(), 1)
        if not grad_norm.isfinite():
            if state.steps < 18:
                raise EDException(name=state.name, age=state.steps, cause=f"{grad_norm=:g}")
            print(f"!!! {grad_norm=:g}, skipping iteration")
            continue
        losses.append(float(loss.detach()))
        decomp_den.append(float(den_t.detach()))
        decomp_vel.append(float(vel_t.detach()))
        decomp_vel_x.append(float(vx_t.detach()))
        decomp_vel_y.append(float(vy_t.detach()))
        state.writer.add_scalar("train iteration loss", loss.detach(), state.steps)
        state.writer.add_scalar("grad norm", grad_norm, state.steps)
        state.optimizer.step()
        state.optimizer.zero_grad()
        state.steps += 1
    loss = sum(losses) / len(losses)
    state.writer.add_scalar("training loss", loss, state.steps)
    if decomp_den:
        ad = sum(decomp_den) / len(decomp_den)
        av = sum(decomp_vel) / len(decomp_vel)
        avx = sum(decomp_vel_x) / len(decomp_vel_x)
        avy = sum(decomp_vel_y) / len(decomp_vel_y)
        vel_scale = cfg.loss_vel_scale if loss_metric == "mean total weighted NLLL" else 1.0
        state.writer.add_scalars(
            "training loss decomposition (epoch mean)",
            {
                "density": ad,
                "velocity": av,
                "velocity x loss_vel_scale": av * vel_scale,
                "vel_est_vx": avx,
                "vel_est_vy": avy,
            },
            state.steps,
        )
        data.set_postfix_str(
            f"mean L={loss:.3f} den={ad:.3f} vel={av:+.3f} velx={avx:+.3f} vely={avy:+.3f}"
        )
        print(
            f"[train epoch {state.epochs}] L_den={ad:.4f} L_vel={av:+.4f} "
            f"L_vel_est_vx={avx:+.4f} L_vel_est_vy={avy:+.4f} L_total={loss:.4f}",
            flush=True,
        )
        save_loss_to_file(state, loss, is_training=True, l_den=ad, l_vel=av, l_vel_x=avx, l_vel_y=avy)
    else:
        data.set_postfix_str(f"mean L={loss:.3f}")
        save_loss_to_file(state, loss, is_training=True)
    state.epochs += 1
    state.writer.add_scalar("epochs", state.epochs, state.steps)
    return loss


def validate(state: PedPredTeacherState, data: DataLoader, loss_metric_name: str):
    state.model.eval()
    model_device = next(state.model.parameters()).device
    with torch.no_grad():
        for name, param in state.model.named_parameters():
            if not param.isfinite().all():
                raise ValueError(f"Parameter {name} has non-finite values.")
            state.writer.add_histogram(f"weights/{name}", param, state.steps)
        losses = []
        all_metrics = defaultdict(list)
        val_decomp_den = []
        val_decomp_vel = []
        val_decomp_vel_x = []
        val_decomp_vel_y = []
        data = tqdm(data, desc=f"{state.name} validating, epoch {state.epochs:,}")
        for i, (input, target) in enumerate(data):
            input = input.to(model_device)
            target = target.to(model_device)
            pred = state.model(input, horizon=target.shape[1])
            metrics = Metrics(pred, target)
            loss = scalar_training_loss(metrics)
            den_v = metrics["mean NLLL density"]
            vel_v = metrics["mean weighted NLLL vel_est"] + metrics["mean weighted NLLL vel_unc"]
            vx_v = metrics["mean weighted NLLL vel_est_vx"]
            vy_v = metrics["mean weighted NLLL vel_est_vy"]
            data.set_postfix_str(_fmt_nlll_postfix(loss, den_v, vel_v, vx_v, vy_v))
            losses.append(loss)
            val_decomp_den.append(float(metrics["mean NLLL density"]))
            val_decomp_vel.append(
                float(metrics["mean weighted NLLL vel_est"] + metrics["mean weighted NLLL vel_unc"])
            )
            val_decomp_vel_x.append(float(metrics["mean weighted NLLL vel_est_vx"]))
            val_decomp_vel_y.append(float(metrics["mean weighted NLLL vel_est_vy"]))
            for metric in metrics.scalars:
                all_metrics[metric].append(metrics[metric])
            if i in range(0, max(len(data), 1), max(len(data) // 8, 1)):
                if state.epochs == 1:
                    state.writer.add_figure(f"plot input/{i}", input[0].plot(), state.steps)
                    state.writer.add_figure(f"plot target/{i}", target[0].plot(), state.steps)
                state.writer.add_figure(f"plot prediction/{i}", pred[0].plot(), state.steps)
        loss = sum(losses) / len(losses)
        state.writer.add_scalar("validation loss", loss, state.steps)
        if val_decomp_den:
            v_den = sum(val_decomp_den) / len(val_decomp_den)
            v_vel = sum(val_decomp_vel) / len(val_decomp_vel)
            v_vel_x = sum(val_decomp_vel_x) / len(val_decomp_vel_x)
            v_vel_y = sum(val_decomp_vel_y) / len(val_decomp_vel_y)
            v_scale = cfg.loss_vel_scale if loss_metric_name == "mean total weighted NLLL" else 1.0
            state.writer.add_scalars(
                "validation loss decomposition (epoch mean)",
                {
                    "density": v_den,
                    "velocity": v_vel,
                    "velocity x loss_vel_scale": v_vel * v_scale,
                    "vel_est_vx": v_vel_x,
                    "vel_est_vy": v_vel_y,
                },
                state.steps,
            )
            print(
                f"[valid epoch {state.epochs}] L_den={v_den:.4f} L_vel={v_vel:+.4f} "
                f"L_vel_est_vx={v_vel_x:+.4f} L_vel_est_vy={v_vel_y:+.4f} L_total={loss:.4f}",
                flush=True,
            )
            save_loss_to_file(
                state,
                loss,
                is_training=False,
                l_den=v_den,
                l_vel=v_vel,
                l_vel_x=v_vel_x,
                l_vel_y=v_vel_y,
            )
        for metric, values in all_metrics.items():
            value = sum(values) / len(values)
            state.writer.add_scalar(f"{metric}", value, state.steps)
        if state.lr_scheduler is not None:
            if isinstance(state.lr_scheduler, ReduceLROnPlateau):
                state.lr_scheduler.step(loss)
            else:
                state.lr_scheduler.step()
            for j, param_group in enumerate(state.optimizer.param_groups):
                state.writer.add_scalar(f"learning rate/{j}", param_group["lr"], state.steps)
        if not val_decomp_den:
            save_loss_to_file(state, loss, is_training=False)
        state.loss = loss
        return loss


def save_loss_to_file(
    state: PedPredTeacherState,
    loss: float,
    is_training: bool = True,
    l_den: float | None = None,
    l_vel: float | None = None,
    l_vel_x: float | None = None,
    l_vel_y: float | None = None,
    l_var: float | None = None,
):
    log_file = os.path.join(state.dir, "loss_log.txt")
    mode = "a" if os.path.exists(log_file) else "w"
    with open(log_file, mode) as f:
        extra = ""
        if l_den is not None:
            extra += f" den={l_den:.6f}"
        if l_vel is not None:
            extra += f" vel={l_vel:+.6f}"
        if l_var is not None:
            extra += f" var={l_var:.6f}"
        if l_vel_x is not None and l_vel_y is not None:
            extra += f" vel_est_vx={l_vel_x:+.6f} vel_est_vy={l_vel_y:+.6f}"
        if is_training:
            f.write(f"Epoch {state.epochs}: Training Loss = {loss}{extra}\n")
        else:
            f.write(f"Epoch {state.epochs}: Validation Loss = {loss}{extra}\n")


def fit(file_glob=None, max_epochs: int = 15):
    print(
        f"[pedpred_teacher_train] objective=nlll lr={cfg.lr} "
        f"weight_decay={getattr(cfg,'weight_decay',0.0)} "
        f"flip_w_prob={getattr(cfg,'flip_w_prob',0.0)} "
        f"early_stop_patience={getattr(cfg,'early_stop_patience',0)}",
        flush=True,
    )
    patience = int(getattr(cfg, "early_stop_patience", 0))
    with cuda_context():
        data = get_teacher_data(
            "train", "valid",
            num_workers=int(os.environ.get("PEDPRED_NUM_WORKERS", "8")),
            pin_memory=True,
            prefetch_factor=int(os.environ.get("PEDPRED_PREFETCH", "4")),
        )
        state = get_trainstate(file_glob, live_cfg=cfg)
        best_val = float("inf")
        epochs_since_best = 0
        try:
            with CatchSignal() as stop:
                while not stop:
                    if state.epochs >= max_epochs:
                        print(f"Reached maximum epochs: {max_epochs}. Stopping training.")
                        break
                    try:
                        train(state, data.train)
                        validate(state, data.valid, loss_metric)
                        cur_val = float(state.loss)
                        if cur_val < best_val - 1e-6:
                            best_val = cur_val
                            epochs_since_best = 0
                        else:
                            epochs_since_best += 1
                            print(
                                f"[early-stop] no-improve {epochs_since_best}/{patience} "
                                f"(cur_val={cur_val:.6f} best={best_val:.6f})",
                                flush=True,
                            )
                            if patience > 0 and epochs_since_best >= patience:
                                print(
                                    f"[early-stop] triggered at epoch {state.epochs} "
                                    f"(no improve for {patience} epochs)",
                                    flush=True,
                                )
                                break
                    except EDException as e:
                        print(e)
                        state = get_trainstate(file_glob, live_cfg=cfg)
        except Exception:
            state.save(err=True)
            raise
        except KeyboardInterrupt:
            state.save(err=True)
            raise
        else:
            state.save()


def main():
    file_glob = None
    max_epochs = 15
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--max-epochs" and i + 1 < len(args):
            max_epochs = int(args[i + 1])
            i += 2
            continue
        if file_glob is None and not arg.startswith("--"):
            file_glob = arg
        i += 1
    fit(file_glob, max_epochs=max_epochs)


if __name__ == "__main__":
    mpl.use("Agg")
    main()
