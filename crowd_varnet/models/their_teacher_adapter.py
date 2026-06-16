"""Adapter: wrap Partial_observation's PedPred3 (nin=1) into our pipeline.

Their PedPred3 expects single-frame input (B, 1, 4, H, W). Our student uses
5-frame history. This wrapper takes the last frame from history and forwards
it through their teacher.

Note: We patched their ConvGRUCell to fix a device bug (torch.zeros without
device kwarg). After the patch, their model runs on GPU normally.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn


def _ensure_partial_observation_path():
    parent = "/scratch/work/zhangx29"
    if parent not in sys.path:
        sys.path.insert(0, parent)
    po_root = "/scratch/work/zhangx29/Partial_observation"
    if po_root not in sys.path:
        sys.path.insert(0, po_root)


class TheirPedPredAdapter(nn.Module):
    """Wraps Partial_observation.PedPred3 for use as frozen prior in CrowdVarNet.

    After patching their ConvGRUCell device bug, the model runs on GPU directly.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        super().__init__()
        _ensure_partial_observation_path()
        from Partial_observation.models import PedPred3

        net = PedPred3()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state)
        net.to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

        # Use object.__setattr__ to prevent nn.Module from registering as submodule
        object.__setattr__(self, '_net', net)
        object.__setattr__(self, '_device', torch.device(device))

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True):
        return iter([])

    def to(self, *args, **kwargs):
        # Move the internal net too
        if args and isinstance(args[0], (str, torch.device)):
            dev = torch.device(args[0])
            self._net.to(dev)
            object.__setattr__(self, '_device', dev)
        return self

    def forward(self, inp, hidden=None, *, horizon=1):
        """Input: tensor [B, T, 4, H, W] or [B, 4, H, W]. Returns pure tensor."""
        if hasattr(inp, "as_tensor"):
            t = inp.as_tensor("density", "vel_mean", "vel_var")
        elif torch.is_tensor(inp):
            t = inp
        else:
            t = torch.as_tensor(inp)

        # Take last frame only (their model uses nin=1)
        if t.dim() == 5:
            t = t[:, -1:].contiguous()
        elif t.dim() == 4:
            t = t.unsqueeze(1)

        # Ensure on correct device
        t = t.to(self._device)

        with torch.no_grad():
            out = self._net(t, horizon=horizon)

        # Extract pure tensor from their GridData output
        if hasattr(out, "as_tensor"):
            tensor_out = out.as_tensor("density", "vel_mean", "vel_var")
        else:
            tensor_out = out

        # Force pure torch.Tensor (their GridData is a tensor subclass)
        if type(tensor_out) is not torch.Tensor:
            tensor_out = torch.empty(tensor_out.shape, dtype=tensor_out.dtype,
                                     device=tensor_out.device).copy_(tensor_out)

        return tensor_out
