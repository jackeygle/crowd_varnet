"""Checkpoint / TensorBoard state for PedPred teacher training (hickle + optional git tag)."""
from __future__ import annotations

from datetime import datetime
from functools import cached_property
from glob import glob
from math import inf
import os
from pathlib import Path
from sys import stderr
import time
from typing import Any, Optional

import petname
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.tensorboard import SummaryWriter

from .pedpred_train_utils import doing


class PedPredTeacherState:
    def __init__(
        self,
        file_glob: Optional[str] = None,
        *,
        model: Optional[Module] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        directory: Optional[os.PathLike] = None,
        live_cfg: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.state: dict = {}
        self.start_time: Optional[float] = None
        self.dir = Path(directory) if directory else Path()
        self._live_cfg = live_cfg

        try:
            import git

            self.repo = git.Repo(path=self.dir, search_parent_directories=True)
        except Exception:
            self.repo = None

        if file_glob is None:
            self.init_state()
        else:
            self.load(file_glob)

    _state_settable = {"steps", "epochs"}
    _state_gettable = {*_state_settable, "name", "commit", "config", "born"}

    def __getattr__(self, key: str):
        if key in self._state_gettable:
            return self.state[key]
        raise AttributeError(key)

    def __setattr__(self, key: str, value):
        if key in self._state_settable:
            self.state[key] = value
        else:
            super().__setattr__(key, value)

    @cached_property
    def writer(self):
        return SummaryWriter(self.dir / "runs" / self.name, purge_step=self.steps + 1)

    @property
    def loss(self):
        return self.state["loss"]

    @loss.setter
    def loss(self, value):
        self.state["loss"] = value
        if value < self.best_loss:
            self.save(best=True)
            self.best_loss = value

    @cached_property
    def best_loss(self):
        file: Path = self._file_name(best=True)
        if file.is_file():
            try:
                return PedPredTeacherState(file).loss
            except Exception:
                pass
        return inf

    def init_state(self):
        self.state = {}
        self.state["name"] = petname.generate()
        print(f"New model: {self.name}")

        if self.repo:
            try:
                tag = self.repo.create_tag(f"model/{self.name}")
                self.state["commit"] = tag.commit.hexsha
            except Exception:
                self.state["commit"] = self.repo.head.commit.hexsha
            if self.repo.is_dirty():
                self.state["commit"] += "-dirty"

        self.state["config"] = vars(self._live_cfg) if self._live_cfg is not None else {}
        self.state["born"] = str(datetime.now())
        self.start_time = time.time()
        self.state["age"] = 0
        self.state["steps"] = 0
        self.state["epochs"] = 0
        self.state["loss"] = inf

    def load(self, file_glob=None):
        file = self._file_glob(file_glob)
        self._load_file(file)
        self._check_git()
        self._check_cfg()
        self._export_state()

    def save(self, *, best=False, err=False):
        self._suffix = self.__class__._suffix
        self._import_state()
        file = self._file_name(best=best, err=err)
        self._save_file(file)

    _suffix = ".hkl"

    def _import_state(self):
        self.state["age"] += time.time() - self.start_time
        self.state["date"] = str(datetime.now())
        self.state["model"] = self.model.state_dict() if self.model else None
        filename = f"{self.name}_train_model_3D.pth"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": self.epochs,
                "loss": self.loss,
            },
            filename,
        )
        self.state["optimizer"] = self.optimizer.state_dict() if self.optimizer else None
        self.state["lr_scheduler"] = self.lr_scheduler.state_dict() if self.lr_scheduler else None

    def _export_state(self):
        if self.model:
            self.model.load_state_dict(self.state["model"])
        if self.optimizer:
            self.optimizer.load_state_dict(self.state["optimizer"])
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(self.state["lr_scheduler"])
        self.start_time = time.time()

    def _load_file(self, file: Path):
        return self._loadsave_file("load", file)

    def _save_file(self, file: Path):
        return self._loadsave_file("save", file)

    def _loadsave_file(self, saveload, file: Path):
        suff = "".join(file.suffixes)
        name = f"_{saveload}{suff.replace('.', '_')}_file"
        try:
            meth = getattr(self, name)
        except AttributeError as e:
            raise NotImplementedError(f"Unknown format, suffix {suff}") from e
        else:
            verbing = dict(load="Loading", save="Saving").get(saveload, "Doing")
            with doing(f"{verbing} {file}"):
                return meth(file)

    def _load_hkl_file(self, file: os.PathLike):
        import hickle

        self.state = hickle.load(file)

    def _save_hkl_file(self, file: os.PathLike):
        import hickle

        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(self.state, dict):
            raise TypeError(f"self.state must be a dict, got {type(self.state)}")
        hickle.dump(self.state, file)

    def _load_pt_pkl_file(self, file: os.PathLike):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = torch.load(file, map_location=device)

    def _save_pt_pkl_file(self, file: os.PathLike):
        torch.save(self.state, file)

    def _file_name(self, *, best: bool = False, err: bool = False) -> Path:
        stamp = "best" if best else f"{self.steps:07}"
        err_tag = "_error" if err else ""
        return self.dir / "checkpoints" / f"{self.name}_{stamp}{err_tag}{self._suffix}"

    def _file_glob(self, file_glob: Optional[str] = None) -> Path:
        if file_glob is None:
            file_glob = ""
        else:
            file_glob = str(file_glob)
        if "/" not in file_glob:
            file_glob = str(self.dir / "checkpoints" / (file_glob + "*" + self._suffix))
        files = glob(file_glob)
        if "_error" not in file_glob:
            filtered = [f for f in files if "_error" not in f]
            if filtered:
                files = filtered
        if files:
            file = max(files, key=os.path.getctime)
        else:
            raise FileNotFoundError(f"No files match {file_glob}")
        return Path(file)

    def _check_git(self):
        if self.repo and self.commit:
            file_hexsha, _, dirty = self.commit.partition("-")
            curr_hexsha = self.repo.head.commit.hexsha
            if file_hexsha != curr_hexsha:
                print(
                    f"Checkpoint commit hash {file_hexsha}",
                    f"   does not match HEAD {curr_hexsha}",
                    sep="\n",
                    file=stderr,
                )
            if dirty:
                print("Checkpoint commit is dirty.", file=stderr)
            if self.repo.is_dirty():
                print("HEAD is dirty.", file=stderr)

    def _check_cfg(self):
        if self._live_cfg is None or "config" not in self.state:
            return
        cfg_dict = vars(self._live_cfg)
        for key in cfg_dict:
            if key in self.config and self.config[key] != cfg_dict[key]:
                print(
                    f"Checkpoint config {key} {self.config[key]}",
                    f"   does not match {key} {cfg_dict[key]}",
                    sep="\n",
                    file=stderr,
                )
