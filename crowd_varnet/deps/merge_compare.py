"""
Merge Teacher vs Crowd-VAE per-step figures for the same assimilation step.

**Preferred:** both runs save ``<run_dir>/step_npz/step_KKK.npz`` (``--dump-step-npz``).
The script redraws Matplotlib from arrays.

**σρ spread color scale** (see ``--spread-scale``):

- **per-row** (default): each row uses its own ``vmax = max(σρ)`` on that row. Spatial
  **patterns** in uncertainty are visible for both methods. Red saturation is **not**
  comparable across Teacher vs Crowd rows (different colorbars).

- **shared**: one ``vmax`` for both rows — same red saturation = same numerical σρ
  **if** both methods have similar magnitude. If Crowd's spread is much larger than
  Teacher's, Teacher's panel can look uniformly pale (compression in the colormap).

**Fallback:** stack pre-rendered ``step_*.png`` (independent colorbars per original PNG).

    python -m crowd_varnet.deps.merge_compare \\
        --compare-root runs_compare_teacher_cvae_17720xxx --merge-mode auto

Crowd-VAE **only** (no teacher folder): regenerate one-row PNGs from
``crowd_vae/step_npz/``, or stack only Crowd raster PNGs::

    python -m crowd_varnet.deps.merge_compare \\
        --compare-root runs_cvae_only_xxx --crowd-only --merge-mode auto
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from .utils_plot import load_step_arrays_npz, plot_generated_matrix_on_ax


def _spread_rho_vmax_pair(d_teacher: dict, d_crowd: dict) -> float:
    r0 = np.asarray(d_teacher["estimated_spread"][0], dtype=np.float64)
    r1 = np.asarray(d_crowd["estimated_spread"][0], dtype=np.float64)
    r0 = np.nan_to_num(r0, nan=0.0, posinf=0.0, neginf=0.0)
    r1 = np.nan_to_num(r1, nan=0.0, posinf=0.0, neginf=0.0)
    return float(max(r0.max(), r1.max(), 1e-12))


def _spread_rho_vmax_one(bundle: dict) -> float:
    r = np.asarray(bundle["estimated_spread"][0], dtype=np.float64)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return float(max(r.max(), 1e-12))


def _draw_method_row(
    fig,
    gs,
    row_idx: int,
    bundle: dict,
    spread_vmax: float,
    panel0_title: str,
    *,
    spread_caption: str,
) -> None:
    po = bundle["partial_obs"]
    tr = bundle["true_state"]
    em = bundle["estimated_mean"]
    es = bundle["estimated_spread"]

    ax0 = fig.add_subplot(gs[row_idx, 0])
    im0 = plot_generated_matrix_on_ax(po, ax0)
    ax0.set_title(panel0_title)

    ax1 = fig.add_subplot(gs[row_idx, 1])
    im1 = plot_generated_matrix_on_ax(tr, ax1)
    ax1.set_title("True state")

    ax2 = fig.add_subplot(gs[row_idx, 2])
    im2 = plot_generated_matrix_on_ax(em, ax2)
    ax2.set_title("Estimated mean")

    ax3 = fig.add_subplot(gs[row_idx, 3])
    rho = np.asarray(es[0], dtype=np.float64).copy()
    rho = np.nan_to_num(rho, nan=0.0)
    smean = float(np.mean(rho))
    slocmax = float(np.max(rho))
    im3 = ax3.imshow(
        rho,
        cmap="Reds",
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=float(spread_vmax),
    )
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title(
        f"σρ spread ({spread_caption})\nmean={smean:.4f}  max={slocmax:.4f}"
    )

    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)


def merge_aligned_npz_crowd_only(
    *,
    path_c: Path,
    out_path: Path,
    dpi: int,
) -> None:
    dc = load_step_arrays_npz(str(path_c))
    step_idx = int(dc["step_idx"])
    v_c = _spread_rho_vmax_one(dc)
    fig = plt.figure(figsize=(21, 5.5))
    gs = gridspec.GridSpec(1, 4, figure=fig, hspace=0.35, wspace=0.2)
    fig.suptitle(
        f"Step {step_idx:03d} — Crowd-VAE + latent EnKF — σρ (row vmax={v_c:.5g})",
        fontsize=11,
        y=0.98,
    )
    _draw_method_row(
        fig,
        gs,
        0,
        dc,
        v_c,
        panel0_title=f"Crowd-VAE · partial obs (step {step_idx})",
        spread_caption=f"row vmax={v_c:.4g}",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def merge_aligned_npz_pair(
    *,
    path_t: Path,
    path_c: Path,
    out_path: Path,
    dpi: int,
    spread_scale: str,
) -> None:
    dt = load_step_arrays_npz(str(path_t))
    dc = load_step_arrays_npz(str(path_c))
    step_idx = int(dt["step_idx"])
    if int(dc["step_idx"]) != step_idx:
        raise ValueError(f"step mismatch {path_t} vs {path_c}")

    if spread_scale == "shared":
        v_t = v_c = _spread_rho_vmax_pair(dt, dc)
        sup = (
            f"Step {step_idx:03d} — σρ: shared vmax={v_t:.5g} for both rows "
            f"(same red ⇔ same σρ only when scales are comparable)"
        )
        cap_t = cap_c = f"shared vmax={v_t:.4g}"
    elif spread_scale == "per-row":
        v_t = _spread_rho_vmax_one(dt)
        v_c = _spread_rho_vmax_one(dc)
        sup = (
            f"Step {step_idx:03d} — σρ: separate color scale per row "
            f"(Teacher vmax={v_t:.5g}, Crowd vmax={v_c:.5g}; compare patterns within each row)"
        )
        cap_t = f"row vmax={v_t:.4g}"
        cap_c = f"row vmax={v_c:.4g}"
    else:
        raise ValueError(spread_scale)

    fig = plt.figure(figsize=(21, 11))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.2)
    fig.suptitle(sup, fontsize=11, y=0.995)

    _draw_method_row(
        fig,
        gs,
        0,
        dt,
        v_t,
        panel0_title=f"Teacher · partial obs (step {step_idx})",
        spread_caption=cap_t,
    )
    _draw_method_row(
        fig,
        gs,
        1,
        dc,
        v_c,
        panel0_title="Crowd-VAE · partial obs",
        spread_caption=cap_c,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def merge_raster_vertical(tp: Path, cp: Path, out_path: Path, dpi: int) -> None:
    img_t = mpimg.imread(tp)
    img_c = mpimg.imread(cp)
    fig_h = 10.0
    fig_w = 20.0
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.12, "height_ratios": [1, 1]},
    )
    axes[0].imshow(img_t)
    axes[0].set_title(
        "Teacher: PedPred + localized EnKF (σρ colorbars not comparable to row below)"
    )
    axes[0].axis("off")
    axes[1].imshow(img_c)
    axes[1].set_title("Crowd-VAE + latent LEnKF")
    axes[1].axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Merge teacher/crowd step figures")
    p.add_argument(
        "--compare-root",
        type=str,
        default=None,
        help="Parent folder containing teacher/ and crowd_vae/ (with --crowd-only: "
        "only crowd_vae/ is used).",
    )
    p.add_argument("--teacher-dir", type=str, default=None)
    p.add_argument("--crowd-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument(
        "--merge-mode",
        type=str,
        choices=("auto", "npz", "raster"),
        default="auto",
        help="npz: redraw from step_npz/; raster: stack PNGs only; "
        "auto: npz when paired step_npz exist else raster.",
    )
    p.add_argument(
        "--spread-scale",
        type=str,
        choices=("per-row", "shared"),
        default="per-row",
        help="σρ colormap: per-row (default) avoids pale Teacher when Crowd vmax ≫ Teacher; "
        "shared uses one vmax for both rows (strict σρ compare if magnitudes are similar). "
        "Ignored when --crowd-only (single row uses its own σρ vmax).",
    )
    p.add_argument(
        "--crowd-only",
        action="store_true",
        help="Skip teacher/: merge only from crowd_vae/ (NPZ redraw or raster copy). "
        "With --compare-root, expects <root>/crowd_vae/ (no teacher/ required).",
    )
    args = p.parse_args()

    crowd_only = bool(args.crowd_only)

    if args.compare_root:
        root = Path(args.compare_root).resolve()
        tdir = root / "teacher"
        cdir = root / "crowd_vae"
        odir = Path(args.out_dir) if args.out_dir else root / "combined"
    else:
        if crowd_only:
            if not args.crowd_dir:
                raise SystemExit("crowd-only: pass --crowd-dir or --compare-root")
            tdir = Path("/nonexistent_placeholder")  # unused
            cdir = Path(args.crowd_dir).resolve()
            odir = Path(args.out_dir).resolve() if args.out_dir else cdir.parent / "combined"
        else:
            if not (args.teacher_dir and args.crowd_dir):
                raise SystemExit("Need --compare-root or both --teacher-dir and --crowd-dir")
            tdir = Path(args.teacher_dir).resolve()
            cdir = Path(args.crowd_dir).resolve()
            odir = Path(args.out_dir).resolve() if args.out_dir else tdir.parent / "combined"

    odir.mkdir(parents=True, exist_ok=True)

    if crowd_only:
        c_npz_all = sorted((cdir / "step_npz").glob("step_*.npz"))
        use_npz = args.merge_mode == "npz"
        if args.merge_mode == "auto":
            use_npz = bool(c_npz_all)

        wrote_npz_c = 0
        if use_npz:
            if not c_npz_all:
                if args.merge_mode == "npz":
                    raise SystemExit(
                        f"[merge] crowd-only merge-mode=npz but no step_*.npz under {cdir}/step_npz"
                    )
            for p_c in c_npz_all:
                merge_aligned_npz_crowd_only(
                    path_c=p_c,
                    out_path=odir / f"{p_c.stem}.png",
                    dpi=int(args.dpi),
                )
                wrote_npz_c += 1
            if wrote_npz_c:
                print(
                    f"[merge] crowd-only aligned redraw → {wrote_npz_c} files in {odir} "
                    "(from crowd_vae/step_npz/)"
                )
                return

        c_pngs = sorted(cdir.glob("step_*.png"))
        if not c_pngs:
            raise SystemExit(f"crowd-only: no step_*.png in {cdir}")

        for cp in c_pngs:
            shutil.copy2(cp, odir / cp.name)
            ncp += 1
        print(f"[merge] crowd-only raster copy → {ncp} files → {odir}")
        return

    t_npz = sorted((tdir / "step_npz").glob("step_*.npz"))
    use_npz = args.merge_mode == "npz"
    if args.merge_mode == "auto":
        use_npz = bool(t_npz) and all((cdir / "step_npz" / p.name).is_file() for p in t_npz)

    wrote_npz = 0
    if use_npz:
        for p_t in t_npz:
            p_c = cdir / "step_npz" / p_t.name
            if not p_c.is_file():
                print(f"[warn] skip {p_t.name}: missing crowd {p_c}")
                continue
            merge_aligned_npz_pair(
                path_t=p_t,
                path_c=p_c,
                out_path=odir / f"{p_t.stem}.png",
                dpi=int(args.dpi),
                spread_scale=args.spread_scale,
            )
            wrote_npz += 1
        if wrote_npz:
            print(
                f"[merge] aligned redraw (spread-scale={args.spread_scale}) → {wrote_npz} files in {odir} "
                "(from step_npz/)"
            )
        elif args.merge_mode == "npz":
            raise SystemExit(
                f"[merge] merge-mode=npz but wrote 0 paired files (check {tdir}/step_npz and {cdir}/step_npz)."
            )

    if wrote_npz > 0:
        return

    if args.merge_mode == "npz":
        raise SystemExit(
            f"[merge] merge-mode=npz requires paired step_*.npz under {tdir}/step_npz and {cdir}/step_npz."
        )

    t_pngs = sorted(tdir.glob("step_*.png"))
    if not t_pngs:
        raise SystemExit(f"No step_*.png in {tdir}")

    n_raster = 0
    for tp in t_pngs:
        cp = cdir / tp.name
        if not cp.is_file():
            print(f"[warn] skip {tp.name}: missing {cp}")
            continue
        merge_raster_vertical(tp, cp, odir / tp.name, int(args.dpi))
        n_raster += 1

    if args.merge_mode == "auto" and n_raster:
        print(
            f"[merge] raster stack → {n_raster} files (use --dump-step-npz on both runs for shared σρ scales)."
        )
    else:
        print(f"[merge] wrote {n_raster} raster files → {odir}")


if __name__ == "__main__":
    main()
