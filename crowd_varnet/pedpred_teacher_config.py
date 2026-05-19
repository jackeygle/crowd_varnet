"""CLI/env config for ``crowd_varnet.pedpred_teacher_train`` (NLLL + pedpred3_gru_mid only)."""
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
    help="loss metric name (kept as a constant for the metric registry)",
    metavar="METRIC",
)
g.add_argument(
    "--loss-vel-scale",
    type=float,
    default=1.0,
    help="Scale velocity NLLL terms in the total loss",
    metavar="S",
)
g.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Adam learning rate",
)
g.add_argument(
    "--weight-decay",
    type=float,
    default=0.0,
    help="AdamW weight_decay; >0 uses AdamW, otherwise Adam (backward compatible)",
)
g.add_argument(
    "--flip-w-prob",
    type=float,
    default=0.0,
    help="Probability of flipping each batch along the W axis (vx negated). Training-only augmentation.",
)
g.add_argument(
    "--early-stop-patience",
    type=int,
    default=0,
    help="Stop after this many epochs without improving val loss; 0 = disabled",
)

cfg, sys.argv[1:] = parser.parse_known_args()

if __name__ == "__main__":
    print(cfg)
    parser.print_values()
    print(f"Remaining args: {sys.argv[1:]}")
