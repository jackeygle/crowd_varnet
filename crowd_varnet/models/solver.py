"""可训练的迭代求解器（VarNet 核心）。

v2 架构改进：
  1. Spatial Self-Attention：每步后加轻量 attention，捕捉全局依赖
  2. 支持 ConvGRU 或 ConvLSTM 作为 RNN cell：
     - ConvGRU + 显式 momentum（默认）
     - ConvLSTM：cell state 天然积累动量，无需显式 momentum（跟 4DVarNet 对齐）

每步输入 [grad, x, x_prior, obs, obs_mask] = 17 通道，
经 RNN + Attention 更新隐状态后通过输出头生成 4 通道 δx。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cost import VariationalCost, clip_crowd_state


class ConvGRUCell(nn.Module):
    """规范 ConvGRU：用两次卷积分别算 (r,z) 与 n。"""

    def __init__(self, in_ch: int, hidden: int, k: int = 3):
        super().__init__()
        self.hidden = hidden
        pad = k // 2
        self.conv_rz = nn.Conv2d(in_ch + hidden, 2 * hidden, kernel_size=k, padding=pad)
        self.conv_n = nn.Conv2d(in_ch + hidden, hidden, kernel_size=k, padding=pad)
        for m in (self.conv_rz, self.conv_n):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        rz = self.conv_rz(torch.cat([x, h], dim=1))
        r, z = rz.chunk(2, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        return (1 - z) * n + z * h


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell：cell state c 天然积累动量（类似 Adam 的一阶矩）。

    跟 4DVarNet 原版对齐：LSTM 的 c 在迭代步之间保持"优化方向记忆"。
    """

    def __init__(self, in_ch: int, hidden: int, k: int = 3):
        super().__init__()
        self.hidden = hidden
        pad = k // 2
        # 一次卷积算 4 个 gate: input, forget, cell, output
        self.conv_gates = nn.Conv2d(in_ch + hidden, 4 * hidden, kernel_size=k, padding=pad)
        nn.init.xavier_uniform_(self.conv_gates.weight)
        nn.init.zeros_(self.conv_gates.bias)
        # Forget gate bias 初始化为 1（鼓励记忆保持）
        with torch.no_grad():
            self.conv_gates.bias[hidden:2*hidden].fill_(1.0)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.conv_gates(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ObsEncoder(nn.Module):
    """Encode sparse observations as a sequence of tokens via Transformer.

    Inputs:
        obs: [B, 4, H, W] observation field (zeros where not observed)
        obs_mask: [B, 1, H, W] binary mask (1=observed)

    Output:
        obs_tokens: [B, N_max, dim] encoded tokens (padded to N_max=H*W)
        token_mask: [B, N_max] valid token mask (True=valid observation)

    Each observed pixel becomes a token: [4 channel values] + [2D pos embedding].
    Transformer self-attention lets each observation "see" all others, building
    a global context that solver can query via cross-attention.
    """

    def __init__(self, dim: int = 128, num_layers: int = 2, num_heads: int = 4,
                 grid_h: int = 36, grid_w: int = 12):
        super().__init__()
        self.dim = dim
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Position embedding: 2D sinusoidal, fixed (not learned)
        # Each pixel (y, x) → 16-dim embedding
        pos_dim = 16
        self.register_buffer("pos_emb", self._build_pos_emb(grid_h, grid_w, pos_dim))

        # Input projection: [4 channels + pos_dim] → dim
        self.input_proj = nn.Linear(4 + pos_dim, dim)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            batch_first=True, activation="relu", dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    @staticmethod
    def _build_pos_emb(H, W, dim):
        """2D sinusoidal position embedding [H, W, dim]."""
        d = dim // 4
        y = torch.arange(H, dtype=torch.float32).unsqueeze(1).expand(H, W)
        x = torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(H, W)
        freqs = torch.exp(-torch.arange(d, dtype=torch.float32) * (math.log(10000.0) / d))
        emb_y = torch.cat([torch.sin(y.unsqueeze(-1) * freqs),
                           torch.cos(y.unsqueeze(-1) * freqs)], dim=-1)
        emb_x = torch.cat([torch.sin(x.unsqueeze(-1) * freqs),
                           torch.cos(x.unsqueeze(-1) * freqs)], dim=-1)
        return torch.cat([emb_y, emb_x], dim=-1)  # [H, W, dim]

    def forward(self, obs: torch.Tensor, obs_mask: torch.Tensor):
        """
        obs: [B, 4, H, W], obs_mask: [B, 1, H, W]
        Returns:
            tokens: [B, N=H*W, dim]
            valid_mask: [B, N] (True=valid observation)
        """
        B, C, H, W = obs.shape
        # Build per-pixel feature: [4 obs values + pos_emb]
        obs_flat = obs.permute(0, 2, 3, 1).reshape(B, H * W, 4)  # [B, N, 4]
        pos = self.pos_emb.reshape(H * W, -1).unsqueeze(0).expand(B, -1, -1)  # [B, N, pos_dim]
        feat = torch.cat([obs_flat, pos], dim=-1)  # [B, N, 4+pos_dim]
        tokens = self.input_proj(feat)  # [B, N, dim]

        # Build src_key_padding_mask: True = padding (unobserved → ignored in self-attn)
        # PyTorch convention: True = mask out
        valid = obs_mask.squeeze(1).reshape(B, H * W) > 0.5  # [B, N], True=observed
        key_padding_mask = ~valid  # True=ignore

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return encoded, valid


class CrossAttention(nn.Module):
    """Cross-attention: solver hidden queries the global observation tokens.

    For each pixel in the spatial hidden state (including unobserved regions),
    attend to all observation tokens to gather global information.
    """

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert query_dim % num_heads == 0

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.norm = nn.GroupNorm(min(8, query_dim), query_dim)

        # Zero-init output for stable residual at init
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor, obs_tokens: torch.Tensor,
                obs_valid: torch.Tensor) -> torch.Tensor:
        """
        h: [B, C, H, W] solver hidden state
        obs_tokens: [B, N, kv_dim] encoded observation tokens
        obs_valid: [B, N] valid mask (True=valid observation)
        Returns h + cross_attn(h, obs_tokens) [residual]
        """
        B, C, H, W = h.shape
        N = obs_tokens.shape[1]

        # Normalize and reshape h → [B, H*W, C]
        h_norm = self.norm(h)
        q_in = h_norm.flatten(2).transpose(1, 2)  # [B, H*W, C]

        q = self.q_proj(q_in)  # [B, H*W, C]
        k = self.k_proj(obs_tokens)  # [B, N, C]
        v = self.v_proj(obs_tokens)  # [B, N, C]

        # Reshape to multi-head
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention with masking on invalid observation tokens
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, heads, H*W, N]

        # Mask: invalid (unobserved) tokens get -inf
        # obs_valid: [B, N] → [B, 1, 1, N]
        attn = attn.masked_fill(~obs_valid[:, None, None, :], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        # Handle case where all keys are masked (no observations at all): nan → 0
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)  # [B, heads, H*W, head_dim]
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)  # [B, H*W, C]
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return h + out


class SpatialAttention(nn.Module):
    """Global spatial self-attention (kept for backward compat / small grids).

    将 [B, C, H, W] reshape 为 [B, HW, C]，做 multi-head attention。
    复杂度 O(N²)，N=H*W。大分辨率（60×96=5760）时请用 WindowSpatialAttention。
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, f"channels={channels} not divisible by num_heads={num_heads}"

        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        N = H * W
        qkv = self.qkv(self.norm(h))
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, C, H, W)
        return h + self.proj(out)


class WindowSpatialAttention(nn.Module):
    """Window-based local spatial self-attention.

    将 [B, C, H, W] 切成 wh×ww 的不重叠窗口，在每个窗口内做 self-attention，
    再拼回来。复杂度 O(N * wh * ww) vs 全局的 O(N²)。

    对于 60×96 网格，window_h=6, window_w=8 → 120 个窗口，每窗口 48 tokens，
    attention 矩阵 48² vs 全局 5760²，快约 120 倍。
    人群交互天然局部，这个设计更符合物理假设。
    """

    def __init__(self, channels: int, num_heads: int = 4,
                 window_h: int = 6, window_w: int = 8, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.window_h = window_h
        self.window_w = window_w
        assert channels % num_heads == 0

        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _to_windows(self, x: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
        """[B, C, Hp, Wp] → [B*nw, ws, C]，ws=wh*ww"""
        B, C, H, W = x.shape
        nh, nwc = H // wh, W // ww
        x = x.reshape(B, C, nh, wh, nwc, ww)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, nh, nwc, wh, ww, C]
        return x.reshape(B * nh * nwc, wh * ww, C)

    def _from_windows(self, x: torch.Tensor, B: int, H: int, W: int, wh: int, ww: int) -> torch.Tensor:
        """[B*nw, ws, C] → [B, C, H, W]"""
        nh, nwc, C = H // wh, W // ww, x.shape[-1]
        x = x.reshape(B, nh, nwc, wh, ww, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, nh, wh, nwc, ww]
        return x.reshape(B, C, H, W)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        wh, ww = self.window_h, self.window_w

        # Pad so H, W are divisible by window size
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        h_pad = F.pad(h, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else h
        Hp, Wp = h_pad.shape[2], h_pad.shape[3]

        qkv = self.qkv(self.norm(h_pad))  # [B, 3C, Hp, Wp]

        # Split into Q/K/V windows
        ws = wh * ww
        nw = (Hp // wh) * (Wp // ww)
        q_win = self._to_windows(qkv[:, :C], wh, ww)          # [B*nw, ws, C]
        k_win = self._to_windows(qkv[:, C:2*C], wh, ww)
        v_win = self._to_windows(qkv[:, 2*C:], wh, ww)

        Bnw = B * nw
        q_win = q_win.reshape(Bnw, ws, self.num_heads, self.head_dim).transpose(1, 2)
        k_win = k_win.reshape(Bnw, ws, self.num_heads, self.head_dim).transpose(1, 2)
        v_win = v_win.reshape(Bnw, ws, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = F.softmax(torch.matmul(q_win, k_win.transpose(-2, -1)) * scale, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_win)                         # [B*nw, heads, ws, head_dim]
        out = out.transpose(1, 2).reshape(Bnw, ws, C)
        out = self._from_windows(out, B, Hp, Wp, wh, ww)        # [B, C, Hp, Wp]

        if pad_h or pad_w:
            out = out[:, :, :H, :W]

        return h + self.proj(out)


class _OutHead(nn.Module):
    """ConvGRU hidden → 4 通道 δ：两层 Conv3x3 + GN + ReLU，再 1x1 出 4。

    Tanh 软限幅避免单步过大。可选 Dropout2d 做 regularization。
    """

    def __init__(self, hidden: int, mid: Optional[int] = None, dropout_p: float = 0.0):
        super().__init__()
        m = mid if mid is not None else hidden
        layers = [
            nn.Conv2d(hidden, m, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, m), num_channels=m),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))
        layers += [
            nn.Conv2d(m, m, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, m), num_channels=m),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))
        layers.append(nn.Conv2d(m, 4, 1))
        self.body = nn.Sequential(*layers)
        # 末层零初始化：训练初期 δ≈0，稳定起步。
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.scale = nn.Parameter(torch.ones(4) * 0.5)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.body(h))
        return delta * self.scale.view(1, 4, 1, 1)


class CrowdVarNetIterativeSolver(nn.Module):
    """ConvGRU/ConvLSTM + Attention 迭代求解器。

    支持两种 RNN cell：
      - "gru": ConvGRU + 显式 momentum（默认，向后兼容）
      - "lstm": ConvLSTM，cell state 天然积累动量，无需显式 momentum（4DVarNet 风格）

    更新公式（4DVarNet 风格）：
      state_update = delta + lr_grad * (step+1)/n_step * grad
      x = x - state_update

    delta 来自学习版迭代器（ConvLSTM/GRU + Attention + OutHead），
    grad 是代价函数的直接梯度。前者负责"探索"，后者负责"保证下降"。

    参数：
      n_iter: 迭代步数（默认 8）
      hidden: RNN 隐藏通道数（默认 256）
      kernel: 卷积核大小（默认 3）
      share_across_iter: 是否所有迭代步共享同一组权重（默认 True）
      dropout_p: 输出头 Dropout2d 概率（默认 0）
      clip_each_step: 每步后是否 clip 状态到物理范围
      use_attention: 是否启用 spatial attention（默认 True）
      attn_heads: attention head 数（默认 4）
      momentum_beta: momentum 系数（0=关闭，仅 gru 模式有效）
      rnn_type: "gru" 或 "lstm"
      lr_grad: 直接梯度项的最大权重（默认 0.0=关闭，0.2=4DVarNet 默认）
    """

    def __init__(
        self,
        n_iter: int = 8,
        hidden: int = 256,
        kernel: int = 3,
        share_across_iter: bool = True,
        dropout_p: float = 0.0,
        clip_each_step: bool = True,
        use_attention: bool = True,
        attn_heads: int = 4,
        momentum_beta: float = 0.5,
        rnn_type: str = "gru",
        lr_grad: float = 0.1,
        use_obs_encoder: bool = False,
        obs_encoder_dim: int = 128,
        obs_encoder_layers: int = 2,
        grid_h: int = 36,
        grid_w: int = 12,
        local_attn: bool = True,
        attn_window_h: int = 6,
        attn_window_w: int = 8,
    ):
        super().__init__()
        self.n_iter = n_iter
        self.hidden = hidden
        self.clip_each_step = clip_each_step
        self.use_attention = use_attention
        self.momentum_beta = momentum_beta
        self.rnn_type = rnn_type.lower()
        # Learnable lr_grad: lets the model discover the right gradient step size.
        # relu-clamped in forward to keep it non-negative.
        self.lr_grad = nn.Parameter(torch.tensor(float(lr_grad)))
        self.use_obs_encoder = bool(use_obs_encoder)
        self.local_attn = local_attn
        self.attn_window_h = attn_window_h
        self.attn_window_w = attn_window_w

        # 输入特征 [grad(4), x(4), x_prior(4), obs(4), obs_mask(1)] = 17
        in_ch = 17
        if share_across_iter:
            if self.rnn_type == "lstm":
                self.rnn_cell = ConvLSTMCell(in_ch=in_ch, hidden=hidden, k=kernel)
            else:
                self.rnn_cell = ConvGRUCell(in_ch=in_ch, hidden=hidden, k=kernel)
            self.out_head = _OutHead(hidden, dropout_p=dropout_p)
            self._rnn_list = None
            self._head_list = None
        else:
            self.rnn_cell = None
            self.out_head = None
            if self.rnn_type == "lstm":
                self._rnn_list = nn.ModuleList(
                    [ConvLSTMCell(in_ch=in_ch, hidden=hidden, k=kernel) for _ in range(n_iter)]
                )
            else:
                self._rnn_list = nn.ModuleList(
                    [ConvGRUCell(in_ch=in_ch, hidden=hidden, k=kernel) for _ in range(n_iter)]
                )
            self._head_list = nn.ModuleList(
                [_OutHead(hidden, dropout_p=dropout_p) for _ in range(n_iter)]
            )
        self.share_across_iter = share_across_iter

        # Spatial attention (shared across iterations)
        if use_attention:
            if local_attn:
                self.attention = WindowSpatialAttention(
                    channels=hidden, num_heads=attn_heads,
                    window_h=attn_window_h, window_w=attn_window_w,
                    dropout=dropout_p,
                )
            else:
                self.attention = SpatialAttention(
                    channels=hidden, num_heads=attn_heads, dropout=dropout_p,
                )
        else:
            self.attention = None

        # Global observation encoder + cross-attention (Perceiver-style)
        if self.use_obs_encoder:
            self.obs_encoder = ObsEncoder(
                dim=obs_encoder_dim, num_layers=obs_encoder_layers,
                num_heads=attn_heads, grid_h=grid_h, grid_w=grid_w,
            )
            self.cross_attn = CrossAttention(
                query_dim=hidden, kv_dim=obs_encoder_dim, num_heads=attn_heads,
            )
        else:
            self.obs_encoder = None
            self.cross_attn = None

        # Learnable momentum (only for GRU mode)
        if self.rnn_type == "gru" and momentum_beta > 0:
            self.momentum_coeff = nn.Parameter(torch.tensor(momentum_beta))
        else:
            self.momentum_coeff = None

    def _step(
        self, k: int, x: torch.Tensor, grad: torch.Tensor,
        x_prior: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor,
        h: torch.Tensor, c: Optional[torch.Tensor] = None,
        obs_tokens: Optional[torch.Tensor] = None,
        obs_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        feat = torch.cat([grad, x, x_prior, obs, obs_mask], dim=1)
        if self.share_across_iter:
            if self.rnn_type == "lstm":
                assert c is not None
                h, c = self.rnn_cell(feat, h, c)
            else:
                h = self.rnn_cell(feat, h)
            if self.attention is not None:
                h = self.attention(h)
            if self.cross_attn is not None and obs_tokens is not None:
                h = self.cross_attn(h, obs_tokens, obs_valid)
            delta = self.out_head(h)
        else:
            assert self._rnn_list is not None and self._head_list is not None
            if self.rnn_type == "lstm":
                assert c is not None
                h, c = self._rnn_list[k](feat, h, c)
            else:
                h = self._rnn_list[k](feat, h)
            if self.attention is not None:
                h = self.attention(h)
            if self.cross_attn is not None and obs_tokens is not None:
                h = self.cross_attn(h, obs_tokens, obs_valid)
            delta = self._head_list[k](h)
        return delta, h, c

    def forward(
        self,
        x_init: torch.Tensor,
        cost_fn: VariationalCost,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        x_prior: torch.Tensor,
    ) -> torch.Tensor:
        x = x_init.clone().detach().requires_grad_(True)
        B, C, H, W = x.shape
        h = torch.zeros(B, self.hidden, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h) if self.rnn_type == "lstm" else None

        # Encode observations once (if obs_encoder enabled)
        obs_tokens = None
        obs_valid = None
        if self.obs_encoder is not None:
            obs_tokens, obs_valid = self.obs_encoder(obs, obs_mask)

        # Momentum buffer (GRU mode only)
        momentum = torch.zeros_like(x_init) if self.momentum_coeff is not None else None

        for k in range(self.n_iter):
            cost, _, _ = cost_fn(x, obs, obs_mask, x_prior)
            grad = torch.autograd.grad(cost, x, create_graph=True)[0]

            delta, h, c = self._step(
                k, x, grad, x_prior, obs, obs_mask, h, c,
                obs_tokens=obs_tokens, obs_valid=obs_valid,
            )

            # 4DVarNet-style update: delta + direct gradient term.
            # lr_grad is a learnable parameter (relu-clamped to stay non-negative).
            # The gradient is L2-normalised per batch item so asymmetric prior
            # weighting (0.05 in unobserved cells) doesn't suppress it to near-zero.
            lr_grad_val = F.relu(self.lr_grad)
            if lr_grad_val > 0:
                grad_norm = (
                    grad.flatten(1).norm(dim=1, keepdim=True)
                    .view(B, 1, 1, 1)
                    .clamp(min=1e-8)
                )
                grad_weight = lr_grad_val * (k + 1) / self.n_iter
                state_update = delta + grad_weight * (grad / grad_norm)
            else:
                state_update = delta

            # Apply update
            if momentum is not None and self.momentum_coeff is not None:
                # GRU + explicit momentum
                beta = torch.sigmoid(self.momentum_coeff)
                momentum = beta * momentum + (1.0 - beta) * state_update
                x = x - momentum
            else:
                # LSTM mode: no explicit momentum (cell state handles it)
                x = x - state_update

            if self.clip_each_step:
                x = clip_crowd_state(x)
            x = x.requires_grad_(True)

        return x
