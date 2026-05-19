"""CLI/env config for ``crowd_varnet.pedpred_teacher_train`` (PedPred_Optimized teacher on ATC)."""
import sys

from configargparse import ArgumentDefaultsHelpFormatter, ArgumentParser

parser = ArgumentParser(
    formatter_class=ArgumentDefaultsHelpFormatter,
    auto_env_var_prefix="PEDPRED_",
    default_config_files=["base.cfg", "machine.cfg"],
    args_for_setting_config_path=("-c", "--config-file"),
    args_for_writing_out_config_file=("-w", "--write-config-file"),
)

g = parser.add_argument_group("Dataset")
g.add_argument(
    "-d",
    "--dataset",
    default="atc:corridor",
    help="dataset name and grid tag (only atc:* is supported here)",
    metavar="DATASET[:GRID]",
)
g.add_argument(
    "--resolution",
    type=float,
    default=1.0,
    help="spatial resolution in meters",
    metavar="RES",
)
g.add_argument(
    "--period",
    type=float,
    default=1.0,
    help="temporal resolution in seconds",
    metavar="D",
)
g.add_argument(
    "--kernel",
    default="tri",
    help="discretisation kernel shape",
    choices=("rect", "tri", "hann"),
    metavar="K",
)

g = parser.add_argument_group("Training")
g.add_argument(
    "-b",
    "--batch",
    type=int,
    default=50,
    help="batch size",
    metavar="N",
)
g.add_argument("--nin", type=int, default=5, help="input steps", metavar="N")
g.add_argument("--nout", type=int, default=5, help="output steps", metavar="N")
g.add_argument(
    "--loss",
    default="mean total weighted NLLL",
    help="loss metric name",
    metavar="METRIC",
)
g.add_argument(
    "--loss-vel-scale",
    type=float,
    default=1.0,
    help="When --loss is 'mean total weighted NLLL', scale velocity NLLL terms",
    metavar="S",
)
g.add_argument(
    "--objective",
    choices=("nlll", "physics_mse", "mse_3ch"),
    default="nlll",
    help="nlll: Metrics NLLL (default). physics_mse: masked MSE on logdensity/vel/logvar. "
         "mse_3ch: MSE on logdensity + velocity, var 通道完全忽略（最稳定）",
)
g.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Adam learning rate",
)
g.add_argument(
    "--physics-density-threshold",
    type=float,
    default=-2.0,
    help="Mask: keep target pixels where first channel (log-density if using log tensor) > this value",
)
g.add_argument("--physics-w-den", type=float, default=10.0, help="physics_mse: weight on global density MSE")
g.add_argument("--physics-w-vel", type=float, default=1.0, help="physics_mse: weight on masked vel MSE")
g.add_argument("--physics-w-var", type=float, default=0.5, help="physics_mse: weight on masked logvar MSE")
g.add_argument("--physics-w-cos", type=float, default=0.0,
               help="velocity direction cos loss weight: w_cos * (1 - cos(pred_v, targ_v)). "
                    "Applies to mse_3ch and nlll objectives.")
g.add_argument("--physics-w-vx-scale", type=float, default=1.0,
               help="NLLL: 单独缩放 vx (vel_est_vx) 项的权重。1.0=不变。")
g.add_argument("--physics-w-vy-scale", type=float, default=1.0,
               help="NLLL: 单独缩放 vy (vel_est_vy) 项的权重。1.0=不变。")
g.add_argument("--weight-decay", type=float, default=0.0,
               help="AdamW weight_decay；>0 时使用 AdamW，否则保持 Adam（向后兼容）")
g.add_argument("--flip-w-prob", type=float, default=0.0,
               help="训练时以此概率沿 W 轴翻转每个 batch（vx 取反）。仅训练增强")
g.add_argument("--early-stop-patience", type=int, default=0,
               help="val 多少 epoch 不刷 best 后早停；0 = 不早停")

cfg, sys.argv[1:] = parser.parse_known_args()

if __name__ == "__main__":
    print(cfg)
    parser.print_values()
    print(f"Remaining args: {sys.argv[1:]}")
