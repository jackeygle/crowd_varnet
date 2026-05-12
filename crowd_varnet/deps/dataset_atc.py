"""
ATC corridor grid-cache dataloaders (same layout as ``partial_observation_experiments.train.get_data`` ATC branch).

默认数据根目录为仓库内 ``data/ATC``（未设置 ``PEDPRED_ATC_DATA_DIR`` 时）。与教师对齐时用同一套
``PEDPRED_ATC_SUBSET`` / ``PEDPRED_RESOLUTION`` / ``PEDPRED_PERIOD`` / ``PEDPRED_KERNEL``（见 ``sbatch/inc_atc_data_env.sh``）。
"""
from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path
from typing import Optional, Union

import h5py
import torch
from numbers import Number
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .grid_data import GridData


def _default_atc_data_dir() -> Path:
    """仓库根目录下 ``data/ATC``（与 ``PEDPRED_ATC_DATA_DIR`` 未设置时一致）。"""
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "ATC"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    return float(v)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


class SeqDataset(Dataset):
    """(input, target) windows from a base dataset."""

    def __init__(self, dataset: Dataset, input_len: int, target_len: int, step="input_len"):
        self.dataset = dataset
        self.input_len = input_len
        self.target_len = target_len
        self.step = getattr(self, step) if isinstance(step, str) else step

    @property
    def seq_len(self):
        return self.input_len + self.target_len

    def __getitem__(self, item):
        if isinstance(item, slice):
            return (self[i] for i in range(*item.indices(len(self))))
        start = item * self.step
        stop = start + self.seq_len
        seq = self.dataset[start:stop]
        inp = seq[: self.input_len]
        target = seq[self.input_len :]
        return inp, target

    def __len__(self):
        return (len(self.dataset) - self.seq_len) // self.step + 1


class CachedGridH5Dataset(Dataset):
    """Read precomputed grid states from cache H5 (``grid``: [N, C, H, W])."""

    def __init__(self, cache_filename: str):
        self.cache_filename = str(cache_filename)
        self.h5 = None
        self.grid = None

    def _ensure_open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.cache_filename, "r", swmr=True, libver="latest")
            self.grid = self.h5["grid"]

    def __getitem__(self, item):
        self._ensure_open()
        if isinstance(item, slice):
            return GridData.stack(tuple(self[i] for i in range(*item.indices(len(self)))))
        grid = torch.from_numpy(self.grid[item])
        return GridData(grid)

    def __len__(self) -> Number:
        self._ensure_open()
        return int(self.grid.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5"] = None
        state["grid"] = None
        return state

    def __del__(self):
        try:
            if getattr(self, "h5", None) is not None:
                self.h5.close()
        except Exception:
            pass


def get_atc_data(
    *mode: str,
    batch: Optional[int] = None,
    nin: Optional[int] = None,
    nout: Optional[int] = None,
    resolution: Optional[float] = None,
    period: Optional[float] = None,
    kernel: Optional[str] = None,
    subset: Optional[str] = None,
    data_dir: Optional[Union[Path, str]] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    validation_num_workers: Optional[int] = None,
):
    """
    Build DataLoaders for ATC ``grid_cache`` HDF5s.

    Env overrides (when a kwarg is ``None``): ``PEDPRED_BATCH``, ``PEDPRED_NIN``, ``PEDPRED_NOUT``,
    ``PEDPRED_RESOLUTION``, ``PEDPRED_PERIOD``, ``PEDPRED_KERNEL``, ``PEDPRED_ATC_SUBSET``,
    ``PEDPRED_ATC_DATA_DIR``.
    """
    mode = mode or ("train", "valid")
    batch = _env_int("PEDPRED_BATCH", 16) if batch is None else batch
    nin = _env_int("PEDPRED_NIN", 5) if nin is None else nin
    nout = _env_int("PEDPRED_NOUT", 1) if nout is None else nout
    resolution = _env_float("PEDPRED_RESOLUTION", 1.0) if resolution is None else resolution
    period = _env_float("PEDPRED_PERIOD", 1.0) if period is None else period
    kernel = _env_str("PEDPRED_KERNEL", "tri") if kernel is None else kernel
    if subset is None:
        subset = _env_str("PEDPRED_ATC_SUBSET", "corridor")
    if not subset:
        subset = "corridor"

    if data_dir is None:
        dd = os.environ.get("PEDPRED_ATC_DATA_DIR", "").strip()
        data_dir = Path(dd) if dd else _default_atc_data_dir()
    else:
        data_dir = Path(data_dir)

    val_nw = num_workers if validation_num_workers is None else validation_num_workers
    r = 1 / resolution
    r = int(r) if float(r).is_integer() else None
    if r is None:
        raise ValueError(f"resolution={resolution!r} must yield integer grid scale factor")

    cache_dir = data_dir / "grid_cache"
    all_readable = [
        "atc-20121024.h5",
        "atc-20121114.h5",
        "atc-20121128.h5",
        "atc-20121219.h5",
        "atc-20130213.h5",
        "atc-20130424.h5",
    ]
    file_splits = {
        "train": all_readable[:4],
        "valid": [all_readable[4]],
        "test": [all_readable[5]],
    }

    def _cache_path(file_name: str) -> Path:
        stem = Path(file_name).stem
        subset_tag = subset or "default"
        kernel_tag = str(kernel).replace(":", "_")
        res_tag = str(resolution).replace(".", "p")
        period_tag = str(period).replace(".", "p")
        return cache_dir / f"{stem}_{subset_tag}_r{res_tag}_p{period_tag}_k{kernel_tag}.h5"

    def _filter_files_with_cache(files: list[str], split_name: str) -> list[str]:
        out: list[str] = []
        for file in files:
            cp = _cache_path(file)
            if cp.is_file():
                out.append(file)
            else:
                print(
                    f"[ATC dataset] skip (no grid cache): {file}\n"
                    f"  expected: {cp}\n"
                    f"  cfg: subset={subset!r} resolution={resolution} period={period} kernel={kernel!r}",
                    flush=True,
                )
        if not out:
            raise RuntimeError(
                f"ATC split {split_name!r}: no usable grid_cache under {cache_dir}. "
                f"Build caches or set PEDPRED_ATC_DATA_DIR / resolution / period / kernel."
            )
        return out

    data: dict[str, DataLoader] = {}
    for m in mode:
        if m not in file_splits:
            continue
        cached_only = _filter_files_with_cache(file_splits[m], split_name=m)
        nw = num_workers if m == "train" else val_nw
        data[m] = DataLoader(
            ConcatDataset(
                [
                    SeqDataset(
                        CachedGridH5Dataset(str(_cache_path(file))),
                        nin,
                        nout,
                    )
                    for file in cached_only
                ]
            ),
            batch,
            shuffle=(m == "train"),
            generator=torch.default_generator,
            num_workers=nw,
            pin_memory=pin_memory,
            drop_last=(drop_last and m == "train"),
            prefetch_factor=prefetch_factor if nw > 0 else None,
            persistent_workers=(nw > 0),
        )

    DataLoaderSet = namedtuple("DataLoaderSet", list(data.keys()))
    out = DataLoaderSet(**data)
    if len(out) == 1:
        return out[0]
    return out
