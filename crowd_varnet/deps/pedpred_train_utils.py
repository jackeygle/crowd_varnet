"""Minimal training helpers for PedPred teacher (cuda default dtype, tqdm, SIGINT)."""
from __future__ import annotations

from contextlib import contextmanager
from signal import SIGINT, signal, strsignal
from typing import Iterator

import torch


@contextmanager
def cuda_context(cuda: bool | None = None) -> Iterator[None]:
    if cuda is None:
        cuda = torch.cuda.is_available()
    old_tensor_type = torch.cuda.FloatTensor if torch.tensor(0).is_cuda else torch.FloatTensor
    old_generator = torch.default_generator
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    torch.default_generator = torch.Generator("cuda" if cuda else "cpu")
    try:
        yield
    finally:
        torch.set_default_tensor_type(old_tensor_type)
        torch.default_generator = old_generator


@contextmanager
def doing(doing_str: str = "Doing", done_str: str = "Done", fail_str: str = "Failed") -> Iterator[None]:
    print(f"{doing_str}...")
    try:
        yield
    except Exception:
        print(f"...{fail_str}!", flush=True)
        raise
    else:
        print(f"...{done_str}.", flush=True)


class CatchSignal:
    def __init__(self, signum: int = SIGINT):
        self.signal = signum
        self.count = 0
        self.prev_handler = None

    def handler(self, signum, frame):
        assert signum == self.signal
        self.count += 1
        print(f"Caught signal {strsignal(signum)} ({self.count} time{'' if self.count == 1 else 's'})", flush=True)
        if self.count >= 10:
            print("My patience is exhausted.", flush=True)
            self.prev_handler(signum, frame)

    def __enter__(self):
        self.prev_handler = signal(self.signal, self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal(self.signal, self.prev_handler)
        return None

    def __bool__(self) -> bool:
        return self.count > 0
