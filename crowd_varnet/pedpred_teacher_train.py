"""
PedPred teacher (ATC) training: NLLL (``Metrics``) or masked MSE (``physics_informed_loss``).

Examples::

    python -u -m crowd_varnet.pedpred_teacher_train [--max-epochs N] [RESUME_GLOB]

    PEDPRED_OBJECTIVE=physics_mse PEDPRED_LR=1e-4 python -u -m crowd_varnet.pedpred_teacher_train ...
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
from .deps.pedpred_metrics import Metrics
from .deps.pedpred_models import PedPred_v3, physics_informed_loss
from .deps.pedpred_train_utils import CatchSignal, cuda_context
from .deps.pedpred_training_state import PedPredTeacherState
from .pedpred_teacher_config import cfg

loss_metric = cfg.loss


def _use_physics() -> bool:
    return cfg.objective == "physics_mse"


def batch_physics_loss(pred, target):
    pred_t = pred.as_tensor("logdensity", "vel_mean", "vel_logvar")
    target_t = target.as_tensor("logdensity", "vel_mean", "vel_logvar")
    return physics_informed_loss(
        pred_t,
        target_t,
        cfg.physics_density_threshold,
        w_den=cfg.physics_w_den,
        w_vel=cfg.physics_w_vel,
    )


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


def _fmt_physics_postfix(
    loss, mse_den: float, mse_vel: float, mse_vx: float, mse_vy: float, mse_cont: float
) -> str:
    L = float(loss.detach())
    return (
        f"L={L:.4f} den={mse_den:.4f} vel={mse_vel:.4f} "
        f"vx={mse_vx:.4f} vy={mse_vy:.4f} cont={mse_cont:.4f}"
    )


def _fmt_nlll_postfix(loss, den, vel, vel_x, vel_y) -> str:
    """L_vel = vel_est+vel_unc (weighted mean); L_velx/L_vely split only the vel_est part (ch0/ch1 of vel_mean)."""
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
    model = PedPred_v3()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), amsgrad=True)
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
    decomp_cont = []
    decomp_phy_vx = []
    decomp_phy_vy = []
    data = tqdm(data, desc=f"{state.name} training, epoch {state.epochs:,}")
    for _, (input, target) in enumerate(data):
        input = input.to(model_device)
        target = target.to(model_device)
        pred = state.model(input, horizon=target.shape[1])
        if _use_physics():
            loss, mse_d, mse_v, mse_vx, mse_vy, mse_cont = batch_physics_loss(pred, target)
            data.set_postfix_str(_fmt_physics_postfix(loss, mse_d, mse_v, mse_vx, mse_vy, mse_cont))
        else:
            loss, den_t, vel_t, vx_t, vy_t = batch_loss_and_decomp(pred, target)
            data.set_postfix_str(_fmt_nlll_postfix(loss, den_t, vel_t, vx_t, vy_t))
        if not loss.isfinite():
            if state.steps < 18:
                raise EDException(name=state.name, age=state.steps, cause=f"{loss=:g}")
            print(f"!!! {loss=:g}, skipping iteration")
            continue
        loss.backward()
        if _use_physics():
            grad_norm = clip_grad_norm_(state.model.parameters(), max_norm=1.0)
        else:
            clip_grad_value_(state.model.parameters(), 1e3)
            grad_norm = clip_grad_norm_(state.model.parameters(), 1)
        if not grad_norm.isfinite():
            if state.steps < 18:
                raise EDException(name=state.name, age=state.steps, cause=f"{grad_norm=:g}")
            print(f"!!! {grad_norm=:g}, skipping iteration")
            continue
        losses.append(float(loss.detach()))
        if _use_physics():
            decomp_den.append(mse_d)
            decomp_vel.append(mse_v)
            decomp_cont.append(mse_cont)
            decomp_phy_vx.append(mse_vx)
            decomp_phy_vy.append(mse_vy)
        else:
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
    ad = av = avx = avy = avr = apvx = apvy = None
    if decomp_den:
        ad = sum(decomp_den) / len(decomp_den)
        av = sum(decomp_vel) / len(decomp_vel)
        if _use_physics():
            acont = sum(decomp_cont) / len(decomp_cont)
            apvx = sum(decomp_phy_vx) / len(decomp_phy_vx)
            apvy = sum(decomp_phy_vy) / len(decomp_phy_vy)
            state.writer.add_scalars(
                "training loss decomposition (epoch mean)",
                {
                    "mse_density": ad,
                    "mse_velocity_masked": av,
                    "mse_velocity_vx_masked": apvx,
                    "mse_velocity_vy_masked": apvy,
                    "continuity": acont,
                },
                state.steps,
            )
            data.set_postfix_str(
                f"mean L={loss:.4f} den={ad:.4f} vel={av:.4f} vx={apvx:.4f} vy={apvy:.4f} cont={acont:.4f}"
            )
            print(
                f"[train epoch {state.epochs}] objective=physics_mse mse_den={ad:.4f} mse_vel={av:.4f} "
                f"mse_vx={apvx:.4f} mse_vy={apvy:.4f} mse_cont={acont:.4f} L_total={loss:.4f}",
                flush=True,
            )
            save_loss_to_file(
                state,
                loss,
                is_training=True,
                l_den=ad,
                l_vel=av,
                l_vel_x=None,
                l_vel_y=None,
                l_var=avr,
                phy_mse_vx=apvx,
                phy_mse_vy=apvy,
            )
        else:
            vel_scale = cfg.loss_vel_scale if loss_metric == "mean total weighted NLLL" else 1.0
            avx = sum(decomp_vel_x) / len(decomp_vel_x)
            avy = sum(decomp_vel_y) / len(decomp_vel_y)
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
            data.set_postfix_str(f"mean L={loss:.3f} den={ad:.3f} vel={av:+.3f} velx={avx:+.3f} vely={avy:+.3f}")
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
        val_decomp_cont = []
        val_decomp_phy_vx = []
        val_decomp_phy_vy = []
        data = tqdm(data, desc=f"{state.name} validating, epoch {state.epochs:,}")
        for i, (input, target) in enumerate(data):
            input = input.to(model_device)
            target = target.to(model_device)
            pred = state.model(input, horizon=target.shape[1])
            if _use_physics():
                loss, mse_d, mse_v, mse_vx, mse_vy, mse_cont = batch_physics_loss(pred, target)
                data.set_postfix_str(_fmt_physics_postfix(loss, mse_d, mse_v, mse_vx, mse_vy, mse_cont))
                losses.append(loss)
                val_decomp_den.append(mse_d)
                val_decomp_vel.append(mse_v)
                val_decomp_cont.append(mse_cont)
                val_decomp_phy_vx.append(mse_vx)
                val_decomp_phy_vy.append(mse_vy)
            else:
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
        v_den = v_vel = v_vel_x = v_vel_y = v_var = vpvx = vpvy = None
        if val_decomp_den:
            v_den = sum(val_decomp_den) / len(val_decomp_den)
            v_vel = sum(val_decomp_vel) / len(val_decomp_vel)
            if _use_physics():
                v_cont = sum(val_decomp_cont) / len(val_decomp_cont)
                vpvx = sum(val_decomp_phy_vx) / len(val_decomp_phy_vx)
                vpvy = sum(val_decomp_phy_vy) / len(val_decomp_phy_vy)
                state.writer.add_scalars(
                    "validation loss decomposition (epoch mean)",
                    {
                        "mse_density": v_den,
                        "mse_velocity_masked": v_vel,
                        "mse_velocity_vx_masked": vpvx,
                        "mse_velocity_vy_masked": vpvy,
                        "continuity": v_cont,
                    },
                    state.steps,
                )
                print(
                    f"[valid epoch {state.epochs}] objective=physics_mse mse_den={v_den:.4f} mse_vel={v_vel:.4f} "
                    f"mse_vx={vpvx:.4f} mse_vy={vpvy:.4f} mse_cont={v_cont:.4f} L_total={loss:.4f}",
                    flush=True,
                )
                save_loss_to_file(
                    state,
                    loss,
                    is_training=False,
                    l_den=v_den,
                    l_vel=v_vel,
                    l_vel_x=None,
                    l_vel_y=None,
                    l_var=v_var,
                    phy_mse_vx=vpvx,
                    phy_mse_vy=vpvy,
                )
            else:
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
        if not _use_physics():
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
    phy_mse_vx: float | None = None,
    phy_mse_vy: float | None = None,
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
        if phy_mse_vx is not None and phy_mse_vy is not None:
            extra += f" phy_mse_vx={phy_mse_vx:.6f} phy_mse_vy={phy_mse_vy:.6f}"
        if l_vel_x is not None and l_vel_y is not None:
            extra += f" vel_est_vx={l_vel_x:+.6f} vel_est_vy={l_vel_y:+.6f}"
        if is_training:
            f.write(f"Epoch {state.epochs}: Training Loss = {loss}{extra}\n")
        else:
            f.write(f"Epoch {state.epochs}: Validation Loss = {loss}{extra}\n")


def fit(file_glob=None, max_epochs: int = 15):
    print(
        f"[pedpred_teacher_train] objective={cfg.objective} lr={cfg.lr} "
        f"(physics threshold={cfg.physics_density_threshold} w_den/w_vel/w_var="
        f"{cfg.physics_w_den}/{cfg.physics_w_vel}/{cfg.physics_w_var})",
        flush=True,
    )
    with cuda_context():
        data = get_teacher_data("train", "valid", pin_memory=False)
        state = get_trainstate(file_glob, live_cfg=cfg)
        try:
            with CatchSignal() as stop:
                while not stop:
                    if state.epochs >= max_epochs:
                        print(f"Reached maximum epochs: {max_epochs}. Stopping training.")
                        break
                    try:
                        train(state, data.train)
                        validate(state, data.valid, loss_metric)
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
