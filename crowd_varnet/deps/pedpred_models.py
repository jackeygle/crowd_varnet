from abc import ABC, abstractmethod
from functools import lru_cache
from math import ceil, floor
from typing import Any, Optional, Union, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (Conv2d, ConvTranspose2d, LeakyReLU, Module,
                      Sequential, Sigmoid, Tanh)
from torch.nn.modules.conv import _ConvNd

from .grid_data import GridData
from .tools_mini import Exporter, seq_to_seq

export = Exporter()


# ==========================================
# 基础 RNN 与卷积块定义
# ==========================================

class AbstractRNN(Module, ABC):
    @abstractmethod
    def forward(self, input: Tensor, hidden: Optional[Any] = None) -> tuple[Tensor, Any]:
        pass

AbstractRNN.register(torch.nn.RNNBase)


@AbstractRNN.register
class RNNCell(torch.nn.RNNCell):
    def forward(self, input, hidden=None):
        hidden = super().forward(input, hidden)
        return hidden, hidden


@AbstractRNN.register
class LSTMCell(torch.nn.LSTMCell):
    def forward(self, input, hidden=None):
        hidden, cell = super().forward(input, hidden)
        return hidden, (hidden, cell)


@AbstractRNN.register
class GRUCell(torch.nn.GRUCell):
    def forward(self, input, hidden=None):
        hidden = super().forward(input, hidden)
        return hidden, hidden


@seq_to_seq
def pad_same(k, *, i=0, s=1, d=1, o=None):
    assert s == 1 or i
    p = ceil(((i - 1) * (s - 1) + d * (k - 1)) / 2)
    assert (o or i) == floor(((i - 1) + 2 * p - d * (k - 1)) / s) + 1
    return p


Conv = TypeVar('Conv')


# ==========================================
# ConvGRUCell（保留，Conv1d/3d 仍可用）
# ==========================================

@AbstractRNN.register
class ConvGRUCell(Module):
    def __init__(self,
                 ConvNd: Union[_ConvNd, Module],
                 in_channels: int, out_channels: int,
                 *,
                 kernel_size=None, in_kernel_size=None, hidden_kernel_size=None,
                 activation=Tanh(), gate_activation=Sigmoid()):
        super().__init__()
        self.activation      = activation
        self.gate_activation = gate_activation
        Ci, Ch = in_channels, out_channels

        assert kernel_size is not None or (in_kernel_size is not None and hidden_kernel_size is not None)
        Ki = kernel_size if kernel_size is not None else in_kernel_size
        Kh = kernel_size if kernel_size is not None else hidden_kernel_size
        if in_kernel_size is not None:    Ki = in_kernel_size
        if hidden_kernel_size is not None: Kh = hidden_kernel_size

        Cg = 3 * Ch
        self.conv_i = ConvNd(Ci, Cg, Ki, padding=pad_same(Ki)) if in_channels else None
        self.conv_h = ConvNd(Ch, Cg, Kh, padding=pad_same(Kh))

    def forward(self, input: Tensor, hidden: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        assert (input is not None) or (hidden is not None)

        if input is not None:
            ci = self.conv_i(input)
        else:
            d  = hidden.ndim - 2
            ci = torch.zeros(1, self.conv_h.out_channels, *(1,) * d, device=hidden.device)

        if hidden is not None:
            ch = self.conv_h(hidden)
        else:
            d  = input.ndim - 2
            ch = self.conv_h.bias.reshape(1, -1, *(1,) * d)
            hidden = 0

        ri, zi, ni = ci.chunk(3, dim=1)
        rh, zh, nh = ch.chunk(3, dim=1)
        r = self.gate_activation(ri + rh)
        z = self.gate_activation(zi + zh)
        n = self.activation(ni + r * nh)
        hidden_out = (1 - z) * n + z * hidden
        return hidden_out, hidden_out


class Conv1dGRU(ConvGRUCell):
    def __init__(self, *args, **kwargs):
        super().__init__(torch.nn.Conv1d, *args, **kwargs)


class Conv3dGRU(ConvGRUCell):
    def __init__(self, *args, **kwargs):
        super().__init__(torch.nn.Conv3d, *args, **kwargs)


# ==========================================
# ConvSTLSTMCell（双记忆）
# H = 空间外观记忆（层内时间流动）
# M = 时序变化记忆（跨层向下流动）
# hidden = (H, M)
# ==========================================

@AbstractRNN.register
class ConvSTLSTMCell(Module):
    """
    双记忆卷积时序单元（PredRNN 风格）
    forward 输入  hidden = (H_prev, M_prev) 或 None
    forward 输出  (H_new, (H_new, M_new))
    """
    def __init__(self,
                 ConvNd: Union[_ConvNd, Module],
                 in_channels: int, out_channels: int,
                 *,
                 kernel_size=None, in_kernel_size=None, hidden_kernel_size=None,
                 activation=Tanh(), gate_activation=Sigmoid()):
        super().__init__()
        self.activation      = activation
        self.gate_activation = gate_activation
        self.out_channels    = out_channels
        Ci, Ch = in_channels, out_channels

        assert kernel_size is not None or (in_kernel_size is not None and hidden_kernel_size is not None)
        Ki = kernel_size if kernel_size is not None else in_kernel_size
        Kh = kernel_size if kernel_size is not None else hidden_kernel_size
        if in_kernel_size is not None:    Ki = in_kernel_size
        if hidden_kernel_size is not None: Kh = hidden_kernel_size

        # 输入卷积：7组门控特征（iH fH gH | iM fM gM | o）
        self.conv_i   = ConvNd(Ci, 7 * Ch, Ki, padding=pad_same(Ki)) if in_channels else None
        # H 记忆卷积：4组（iH fH gH | o_H）
        self.conv_h   = ConvNd(Ch, 4 * Ch, Kh, padding=pad_same(Kh))
        # M 记忆卷积：3组（iM fM gM）
        self.conv_m   = ConvNd(Ch, 3 * Ch, Kh, padding=pad_same(Kh))
        # 输出投影：[C_H; C_M] → Ch
        self.conv_out = ConvNd(2 * Ch, Ch, 1)

    def forward(self,
                input: Optional[Tensor],
                hidden: Optional[tuple] = None) -> tuple[Tensor, tuple]:
        H_prev, M_prev = hidden if hidden is not None else (None, None)
        assert (input is not None) or (H_prev is not None)

        # ── 输入卷积 ─────────────────────────────────────────
        if input is not None:
            xi = self.conv_i(input)
        else:
            d  = H_prev.ndim - 2
            xi = torch.zeros(1, 7 * self.out_channels,
                             *(1,) * d, device=H_prev.device)
        xi_iH, xi_fH, xi_gH, xi_iM, xi_fM, xi_gM, xi_o = xi.chunk(7, dim=1)

        # ── H 路（空间记忆）──────────────────────────────────
        if H_prev is not None:
            xh = self.conv_h(H_prev)
        else:
            d      = input.ndim - 2
            xh     = self.conv_h.bias.reshape(1, -1, *(1,) * d)
            H_prev = torch.zeros_like(xi_iH)
        xh_iH, xh_fH, xh_gH, xh_o = xh.chunk(4, dim=1)

        i_H = self.gate_activation(xi_iH + xh_iH)
        f_H = self.gate_activation(xi_fH + xh_fH)
        g_H = self.activation(xi_gH + xh_gH)
        C_H = f_H * H_prev + i_H * g_H

        # ── M 路（时序记忆，跨层流动）────────────────────────
        if M_prev is not None:
            xm = self.conv_m(M_prev)
        else:
            ref = input if input is not None else H_prev
            xm  = torch.zeros(1, 3 * self.out_channels,
                              *(1,) * (ref.ndim - 2), device=ref.device)
        xm_iM, xm_fM, xm_gM = xm.chunk(3, dim=1)

        i_M = self.gate_activation(xi_iM + xm_iM)
        f_M = self.gate_activation(xi_fM + xm_fM)
        g_M = self.activation(xi_gM + xm_gM)
        M_zero = torch.zeros_like(g_M)
        C_M    = f_M * (M_prev if M_prev is not None else M_zero) + i_M * g_M

        # ── 输出门（融合 H 和 M）─────────────────────────────
        o     = self.gate_activation(xi_o + xh_o)
        H_new = o * self.activation(self.conv_out(torch.cat([C_H, C_M], dim=1)))
        M_new = C_M

        return H_new, (H_new, M_new)


class Conv2dSTLSTM(ConvSTLSTMCell):
    """2D ST-LSTM，直接替换 Conv2dGRU"""
    def __init__(self, *args, **kwargs):
        super().__init__(torch.nn.Conv2d, *args, **kwargs)


# ==========================================
# InfIterable
# ==========================================

@lru_cache
class InfIterable:
    def __init__(self, value): self.value = value
    def __iter__(self): return self
    def __reversed__(self): return self
    def __next__(self): return self.value
    def __getitem__(self, item): return self.value
    def __contains__(self, item): return item == self.value
    def __eq__(self, other): return self.value == other.value

Nones = InfIterable(None)


# ==========================================
# SequentialRNNCell
# 兼容 GRU 单隐状态 / ST-LSTM (H,M) 双隐状态
# ==========================================

@AbstractRNN.register
class SequentialRNNCell(Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.is_rnn = [isinstance(m, AbstractRNN) for m in self]

    def forward(self, input: Tensor, hidden=None):
        output = input
        if hidden is None:
            hidden = Nones
        hidden_iter = iter(hidden)
        hidden_out  = []
        for module, is_rnn in zip(self, self.is_rnn):
            if is_rnn:
                h_in  = next(hidden_iter)
                output, h_out = module(output, h_in)
                hidden_out.append(h_out)
            else:
                output = module(output)
        assert next(hidden_iter, None) is None
        return output, hidden_out


class SequentialRNN(SequentialRNNCell):
    def forward(self, input, hidden=None):
        output = []
        for i in input:
            o, hidden = super().forward(i, hidden)
            if o is not None:
                output.append(o)
        if output:
            output = torch.stack(output)
        return output, hidden


class Encoder(SequentialRNN):
    def forward(self, input, hidden=None):
        all_hidden  = []
        curr_hidden = hidden
        for i in input:
            o, curr_hidden = SequentialRNNCell.forward(self, i, curr_hidden)
            all_hidden.append(curr_hidden)
        return curr_hidden, all_hidden


class Forecaster(SequentialRNN):
    def __init__(self, *args, horizon=None):
        super().__init__(*args)
        self.horizon = horizon

    def forward(self, hidden, *, horizon=None):
        horizon = horizon or self.horizon
        output, hidden = super().forward([None] * horizon, hidden)
        return output, hidden


# ==========================================
# 注意力门控（U-Net 跳跃连接）
# ==========================================

class AttentionGate(Module):
    """过滤编码器跳跃连接中的无关空间特征"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g     = Conv2d(F_g, F_int, kernel_size=1)
        self.W_x     = Conv2d(F_l, F_int, kernel_size=1)
        self.psi     = Conv2d(F_int, 1, kernel_size=1)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        return x * self.sigmoid(self.psi(self.relu(g1 + x1)))


# ==========================================
# 时间注意力聚合（兼容 GRU 单状态 / ST-LSTM 双状态）
# ==========================================

class TemporalAttention(Module):
    """对 Encoder 所有时间步隐状态做 Q-K-V 加权聚合"""
    def __init__(self, channels):
        super().__init__()
        self.query = Conv2d(channels, channels // 4, kernel_size=1)
        self.key   = Conv2d(channels, channels // 4, kernel_size=1)
        self.value = Conv2d(channels, channels, kernel_size=1)

    def forward(self, hidden_seq):
        # ST-LSTM 隐状态是 (H, M)，只取 H 做注意力聚合
        seq     = [h[0] if isinstance(h, tuple) else h for h in hidden_seq]
        T       = len(seq)
        stacked = torch.stack(seq, dim=1)       # [B, T, C, H, W]
        B, _, C, H, W = stacked.shape
        flat    = stacked.view(B * T, C, H, W)

        q = self.query(flat).view(B, T, -1)
        k = self.key(flat).view(B, T, -1)
        v = self.value(flat).view(B, T, -1)

        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5), dim=-1)
        out  = torch.bmm(attn, v)
        return out[:, -1].view(B, C, H, W)


# ==========================================
# EncoderForecaster（含跳跃连接 + 时间注意力）
# ==========================================

class EncoderForecaster(Module):
    def __init__(self, encoder, forecaster, horizon=None,
                 skip_channels=None, use_temporal_attn=True):
        super().__init__()
        self.encoder    = encoder    if isinstance(encoder,    Encoder)    else Encoder(*encoder)
        self.forecaster = forecaster if isinstance(forecaster, Forecaster) else Forecaster(*forecaster)
        self.forecaster.horizon = horizon

        self.attn_gates = None
        if skip_channels:
            self.attn_gates = nn.ModuleList([
                AttentionGate(F_g=ch, F_l=ch, F_int=max(ch // 2, 4))
                for ch in skip_channels
            ])

        self.temporal_attns = None
        if use_temporal_attn and skip_channels:
            self.temporal_attns = nn.ModuleList([
                TemporalAttention(ch) for ch in skip_channels
            ])

    def forward(self, input, hidden=None, *, horizon=None):
        final_hidden, all_hidden = self.encoder(input, hidden)

        if self.temporal_attns is not None:
            num_rnn = len(final_hidden)
            for i in range(num_rnn):
                layer_hidden_seq = [step[i] for step in all_hidden]
                if i < len(self.temporal_attns):
                    ctx = self.temporal_attns[i](layer_hidden_seq)
                    # ST-LSTM：用聚合 H 替换，保留 M 不变
                    if isinstance(final_hidden[i], tuple):
                        _, M = final_hidden[i]
                        final_hidden[i] = (ctx, M)
                    else:
                        final_hidden[i] = ctx

        reversed_hidden = list(reversed(final_hidden))
        output, _ = self.forecaster(reversed_hidden, horizon=horizon)
        return output


# ==========================================
# 物理损失函数
# 4通道，vel_logvar 不参与损失，兼容 GridData
# ==========================================

def spatial_divergence(density: Tensor, velocity: Tensor) -> Tensor:
    """∇·(ρv)，用于连续性方程约束"""
    rho    = density.exp()
    rho_vx = rho * velocity[:, :, 0:1]
    rho_vy = rho * velocity[:, :, 1:2]

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=density.dtype, device=density.device).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(2, 3)

    B, T, _, H, W = rho_vx.shape
    d_dx = F.conv2d(rho_vx.view(B * T, 1, H, W), sobel_x, padding=1)
    d_dy = F.conv2d(rho_vy.view(B * T, 1, H, W), sobel_y, padding=1)
    return (d_dx + d_dy).view(B, T, 1, H, W)


def physics_informed_loss(
    pred_seq,
    target_seq,
    density_threshold: float = -2.0,
    *,
    w_den:  float = 10.0,
    w_vel:  float = 5.0,
    w_dir:  float = 0.5,
    w_cont: float = 1.0,
):
    """
    shape: [B, T, C=4, H, W]
    C=4: logdensity(1) + vel_mean(2) + vel_logvar(1)
    vel_logvar 不参与损失，完全兼容 GridData 4通道格式。
    """
    pred_den, pred_vel, _ = pred_seq.split((1, 2, 1), dim=2)
    targ_den, targ_vel, _ = target_seq.split((1, 2, 1), dim=2)

    # 防止目标 logdensity 中的 -inf 导致梯度爆炸
    targ_den = torch.clamp(targ_den, min=-20.0)

    # 1. 密度损失（实际密度域，防止背景对数误差主导梯度）
    loss_den = F.mse_loss(pred_den.exp(), targ_den.exp())

    # 2. 软密度权重掩码
    weight       = torch.sigmoid((targ_den - density_threshold) * 2.0)
    valid_pixels = weight.sum() + 1e-6

    # 3. 速度 MSE（软权重）
    loss_vel = (F.mse_loss(pred_vel, targ_vel, reduction="none") * weight).sum() / valid_pixels

    # 4. x/y 方向分量损失
    pvx, pvy = pred_vel.split(1, dim=2)
    tvx, tvy = targ_vel.split(1, dim=2)
    loss_vel_x = (F.mse_loss(pvx, tvx, reduction="none") * weight).sum() / valid_pixels
    loss_vel_y = (F.mse_loss(pvy, tvy, reduction="none") * weight).sum() / valid_pixels
    loss_dir   = loss_vel_x + loss_vel_y

    # 5. 连续性方程残差（∂ρ/∂t + ∇·(ρv) ≈ 0）
    div       = spatial_divergence(pred_den, pred_vel)
    dpdt      = pred_den[:, 1:] - pred_den[:, :-1]
    loss_cont = F.mse_loss(dpdt, -div[:, :-1])

    total_loss = (
        w_den  * loss_den  +
        w_vel  * loss_vel  +
        w_dir  * loss_dir  +
        w_cont * loss_cont
    )

    return (
        total_loss,
        loss_den.item(),
        loss_vel.item(),
        loss_vel_x.item(),
        loss_vel_y.item(),
        loss_cont.item(),
    )


# ==========================================
# 最终模型 PedPred_v4
# Conv2dGRU 全部替换为 Conv2dSTLSTM（双记忆结构）
# 4通道，完全兼容 GridData
# ==========================================

@export
class PedPred_v4(EncoderForecaster):
    def __init__(self, *args, **kwargs):
        C = 1 + 2 + 1   # logdensity(1) + vel_mean(2) + vel_logvar(1)

        rnn_kwargs = dict(in_kernel_size=3, hidden_kernel_size=5, activation=LeakyReLU(0.2))

        hid0, hid1, hid2 = 16, 32, 128

        encoder = Encoder(
            Conv2d(C, prev := 16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
            Conv2dSTLSTM(prev, prev := hid0, **rnn_kwargs),
            Conv2d(prev, prev := 32, kernel_size=3, stride=2, padding=1), LeakyReLU(0.2),
            Conv2dSTLSTM(prev, prev := hid1, **rnn_kwargs),
            Conv2d(prev, prev := 64, kernel_size=3, stride=2, padding=1), LeakyReLU(0.2),
            Conv2dSTLSTM(prev, prev := hid2, **rnn_kwargs),
        )

        forecaster = Forecaster(
            Conv2dSTLSTM(None, prev := hid2, **rnn_kwargs),
            ConvTranspose2d(prev, prev := 32, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2),
            Conv2dSTLSTM(prev, prev := hid1, **rnn_kwargs),
            ConvTranspose2d(prev, prev := 16, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2),
            Conv2dSTLSTM(prev, prev := hid0, **rnn_kwargs),
            ConvTranspose2d(prev, prev := 16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
            Conv2d(prev, prev := 16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
            Conv2d(prev, C, kernel_size=1),
        )

        super().__init__(
            encoder, forecaster, *args,
            skip_channels=[hid0, hid1, hid2],
            use_temporal_attn=True,
            **kwargs
        )

    def forward(self, input, hidden=None, *, horizon=None):
        input = GridData(input).as_tensor('density', 'vel_mean', 'vel_var')
        input = input.transpose(0, 1)   # [B,T,C,H,W] -> [T,B,C,H,W]

        output = super().forward(input, hidden, horizon=horizon)

        logdensity, vel_mean, vel_logvar = output.split((1, 2, 1), dim=-3)

        # 物理边界截断
        logdensity = torch.clamp(logdensity, min=-20.0, max=10.0)
        vel_mean   = torch.clamp(vel_mean,   min=-15.0, max=15.0)
        vel_logvar = torch.clamp(vel_logvar, min=-20.0, max=5.0)

        output = GridData(logdensity=logdensity, vel_mean=vel_mean, vel_logvar=vel_logvar)
        output = GridData(output.transpose(1, 0))   # [B,T,C,H,W] <- [T,B,C,H,W]
        return output


 # 接口完全兼容，直接替换
PedPred3   = PedPred_v4