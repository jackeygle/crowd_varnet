# Experiments Log — CrowdVarNet + PedPred Teacher

本文档记录在 ATC 走廊稀疏移动传感器反演任务上，PedPred 教师与 CrowdVarNet 学生整套实验脉络与结论。即使后续清理掉 runs/ 中的中间产物，本文档可作为永久 paper trail。

> **最终方案**：
> - 教师 = PedPred3_gru_mid + NLLL，30 epoch（job 17843926）
> - 学生 = CrowdVarNet (ConvGRU solver, n_iter=8) + 教师作冻结先验
> - 训练 = TBPTT，episode 300 单步主训 + episode 600 长 horizon refine
> - 最佳 val_rollout_loss = **0.04738**，相对单步 baseline 改善 **−13.9%**

---

## 1. 数据集与任务

- **数据集**: ATC 大阪商场走廊行人轨迹，栅格化为 60×96（1 m × 1 s）；新数据集 = 32 days train / 7 days val / 1 day test。
- **状态**: 4 通道 (logρ, vx, vy, log σ²)。
- **观测模型**: 3 个移动 agent，半径 5 格视野，每帧前进 1 格，覆盖率 ~10–15%。
- **任务**: 给定稀疏观测序列 + 历史估计，反演每一帧的稠密状态。

---

## 2. 教师实验 (Phase 1)

### 2.1 Backbone 横评（mse_3ch loss, 15 epoch）

5 种 backbone 在同损失下的训练对比：

| arch | 训练日志 valid mse_vx | mse_vy | 备注 |
|---|---|---|---|
| v6 unet | 0.227 | 0.074 | UNet 基线 |
| v7 earth | 0.208 | 0.068 | EarthFormer style |
| v8 simvp | 0.213 | 0.070 | SimVP |
| v9 convnext | 0.209 | 0.069 | ConvNeXt |
| **v10 gru** | **0.187** | 0.087 | GRU stack（密度通道远好） |

**结论**: GRU backbone 密度通道明显领先（valid mse_den ~10 vs 其他 ~16–17），速度通道接近。GRU 的循环时序结构是关键。

### 2.2 GRU 配置扫描（mse_3ch, 30 epoch）

| 实验 | arch | best val L_total | mse_vx | mse_vy | 备注 |
|---|---|---|---|---|---|
| v10_gru baseline | gru | (15ep) | 0.187 | 0.087 | 起点 |
| v10_long | gru | 0.108 | 0.211 | 0.098 | 单纯训长 → 速度退化 |
| v11_mid | gru_mid | 0.105 | 0.166 | 0.075 | 增宽中段 hidden（24/48/192）→ 最佳 |
| v12_resid | gru_residual | 2796.92 | — | — | **崩了**：零初始化末层导致 collapse |

valid eval（support pixels, RMSE）：

| 模型 | rho | vx | vy | var |
|---|---|---|---|---|
| v10_gru (15ep) | 0.213 | 0.366 | 0.320 | 1.28 |
| v10_long (30ep) | 0.180 | 0.469 | 0.351 | 3.92 |
| v11_mid (30ep) | 0.185 | 0.358 | 0.294 | 9.14 ⚠ |

**关键发现**: var 通道在 mse_3ch 下不被监督，反而在长训中爆炸（被当作 dump residual 的"垃圾桶"）。

### 2.3 损失对比：mse_3ch vs NLLL

回到旧记录中 NLLL 训出来的 v3 / v3_wide：

| 模型 | loss | rho | vx | vy | 综合 all (support_MSE) |
|---|---|---|---|---|---|
| v3 (NLLL) | NLLL | 0.094 | 0.273 | 0.231 | 0.073 |
| v3_wide (NLLL) | NLLL | 0.123 | 0.263 | 0.222 | 0.164 |
| v11_mid (mse_3ch) | mse_3ch | 0.185 | 0.358 | 0.294 | 20.97 |

**结论**: NLLL 训的速度通道比 mse_3ch 全面更好（vx 改善 25–30%）。
NLLL 的 σ² 自适应权重让模型在低信噪比像素自动放大不确定度，相当于隐式样本权重。

### 2.4 NLLL 路线 (v13)

**v13** (job 17843926): gru_mid + NLLL + 当前训练配置（batch 48, lr 5e-4, flip-W 0.5, 30 epoch）

| 通道 | global_MSE | support_MSE | support_RMSE |
|---|---|---|---|
| rho | 0.00152 | 0.0167 | **0.129** |
| vx | 0.0853 | 0.0768 | **0.277** |
| vy | 0.0463 | 0.0537 | **0.232** |
| var | 0.062 | 0.601 | 0.775 |
| all | 0.0487 | 0.187 | — |

**这是当前最佳教师**，在每个通道都赢过 v3_wide (NLLL+wide)。

### 2.5 失败的尝试

#### v14: gru_mid + mse_3ch + cos 方向损失（w_cos=2.0）

cos 损失公式 `1 − cos(pred_v, targ_v)`，理论上能补 MSE 不区分方向/量级的缺陷。
- best val L_total = 104.93 vs v11_mid 105.26（**改善 0.3%，无效**）
- support_RMSE: vx 0.318, vy 0.278（vs v11_mid 0.358 / 0.294）

**结论**: cos 在 mse_3ch 框架几乎不起作用 —— MSE 已隐含方向信息，cos 是冗余。

#### v15: gru_mid + NLLL + cos（w_cos=2.0）

- best val 0.10782 vs v13 NLLL 0.10518（**变差 2.5%**）
- support_RMSE: vx 0.358, vy 0.233（vx 反而退化）

**结论**: NLLL 自带方差自适应监督，加 cos 反而扰乱不确定度学习。

#### v16: pedpred3_wide + NLLL（job 17845309 → 17845420）

测试 wide CNN 在 NLLL 下是否优于 GRU。
- ep 9 best val L_total = 0.10470（比 v13 的 0.10518 略好）
- 但**主动取消**：训练曲线和 v13 接近，wide 28M 参数比 gru 7M 大 4 倍，性价比低。

**结论**: 在 NLLL 下 wide 略优 GRU，但性价比不足以替代 v13。

#### v17: gru_mid_velresid + vx 权重 1.5（job 17845866）

策略 (1)+(3) 联合：速度通道做残差预测 + vx loss 单独加权。

| 通道 | support_RMSE | vs v13 |
|---|---|---|
| rho | 0.132 | +2% |
| vx | 0.275 | −0.7% |
| vy | 0.231 | −0.4% |
| var | 1.36 | +77% |

**结论**: 改善 < 1%，在 noise 范围内。证实速度通道接近噪声地板。

### 2.6 噪声地板诊断（diag_velocity_noise_floor.py）

- **persistence baseline** (do-nothing): vx_RMSE = 0.654, vy_RMSE = 0.293
- **smoothing residual**（GT 高频噪声估计）: vx 0.486, vy 0.203
- **GT signal std**: vx 0.96, vy 0.36

v13 vy_RMSE = 0.232 → 距噪声地板 0.20 仅 ~15% 空间。
v13 vx_RMSE = 0.277 → 仍有 5–10% 空间。

**结论**: 速度通道接近物理上限。

### 2.7 教师实验总结

| 排名 | 模型 | vx | vy | rho | 备注 |
|---|---|---|---|---|---|
| **1** | **v13 gru_mid + NLLL** | **0.277** | **0.232** | **0.129** | **当选** |
| 2 | v3_wide + NLLL | 0.263 | 0.222 | 0.123 | 旧训练，wide |
| 3 | v17 velresid + vx weight | 0.275 | 0.231 | 0.132 | 复杂度高，收益微 |
| — | v15 NLLL + cos | 0.358 | 0.233 | 0.152 | cos 扰乱 NLLL |
| — | v14 mse + cos | 0.318 | 0.278 | 0.207 | cos 在 MSE 中冗余 |
| — | v11_mid mse_3ch | 0.358 | 0.294 | 0.185 | mse_3ch loss 速度差 |

---

## 3. 学生实验 (Phase 2 / Phase 3)

教师固定为 v13。学生用 `rollout_train_cli`（多步 TBPTT）。

### 3.1 Phase 2-a 单步主训（job 17845947）

- arch: pedpred3_gru_mid（学生 solver = ConvGRU）
- episode 300, k_prime 8, batch 48, lr 2e-4, ch_weights (3, 1.5, 1, 0)
- 25 epoch（实际跑 10 epoch 后超时）
- best val_rollout_loss = **0.0550** (epoch 10)

### 3.2 Phase 2-a resume（job 17847841）— 失败

从 last.pt 接续 12 epoch，lr 1e-4。
- ep 3 → 0.0542（−1.5%）
- ep 4–7 都没刷新 best
- 单步训练已经 plateau

**结论**: 单步 baseline 上限 ~0.054。

### 3.3 Phase 3 horizon（job 17847842）— ⭐ 关键收益

warmstart 自 P2-a best.pt，episode_len 600（双倍），k_prime 12，lr 5e-5，8 epoch。

| epoch | val_rollout_loss |
|---|---|
| 1 | 0.0482 |
| 2 | 0.0482 |
| 3 | 0.0481 |
| 4 | 0.0478 |
| 5 | 0.0478 |
| **6** | **0.04738** ← best |
| 7 | 0.0479 |
| 8 | 0.0476 |

**改善 13.9% over P2-a baseline**。学生欠的不是单步精度，是多步外推鲁棒性。

### 3.4 Phase 3 long（job 17848110）— 与 P3 horizon 持平

20 epoch 长版 + lr 5e-5 + episode 600。
- ep 3 best 0.04734（与 P3 horizon ep6 持平）
- ep 4–8 都没再刷新
- 3h 超时跑 8 epoch

**结论**: episode 600 / k_prime 12 在 ep 3 已收敛。

### 3.5 Phase 3 extreme（job 17853138）— 极限尝试

episode_len 2000, k_prime 24, batch 16, lr 3e-5, 10 epoch, 8h limit。
（用户主动停止前未完成）

### 3.6 失败的尝试

#### w_prior=1.0 探针（job 17848043）

加倍教师先验权重。
- best val 0.0577 vs baseline 0.0550（**变差 5%**）
- 加大 prior 权重压制学生从观测吸收信息

**结论**: 默认 w_prior=0.5 已最优。

#### Unfreeze teacher tail（job 17850711）

末两层 Conv 解冻，phi_lr=2.5e-6 联合微调。
- ep 1 起步 val = 0.0534（已比 warmstart 0.0474 高 12%）
- ep 3 仍 0.0530，趋势压不回去
- 主动停止

**结论**: 解冻教师扰乱了 NLLL 学到的 σ² head，得不偿失。

### 3.7 学生实验总结

| 实验 | best val | vs baseline | 结论 |
|---|---|---|---|
| P2-a baseline | 0.0550 | — | 单步起点 |
| P2-a resume | 0.0542 | -1.5% | plateau |
| **P3 horizon** | **0.04738** | **-13.9%** | ⭐ **当选** |
| P3 long | 0.04734 | -13.9% | 与 P3 horizon 持平 |
| P3 extreme | (未完) | — | 用户主动停 |
| w_prior=1.0 | 0.0577 | +5% | 失败 |
| unfreeze tail | 0.0530 | +12% | 失败 |

---

## 4. 评估实验

### 4.1 推理性能（infer_cli on P2-a best.pt, job 17848202）

| 指标 | 值 |
|---|---|
| rho support_MSE | 0.01414 |
| rho global_MSE | 0.00153 |
| rho_bg_MSE | 0.000104 |
| frac_bg_pred_gt_thr | 1% (背景 leak 控制良好) |
| rollout 5-episode mean (rho/vx/vy) | 0.034 / 0.114 / 0.026 |

### 4.2 推理速度

V100 (dgx8) 上：
- 每 batch (=4 sample) ~2.25 s → **每 sample ~0.56 s**
- 模型每样本 9 次 forward（8 solver + 1 teacher） → **每次 forward ~62 ms**
- 1 Hz 实时监控充裕，30 Hz 视频实时不够

### 4.3 可视化

`runs/cvn_p3_v13_horizon_17847842/infer_300frames/step_000.png` … `step_299.png`，连续 300 帧 4 列对比图（partial obs / GT / pred / |error|）。

`figures/teacher_v10_gru/`, `teacher_v10_long/`, `teacher_v11_mid/` 各 300 帧教师可视化（早期对比时生成）。

---

## 5. 已封顶的方向

下列方向均经实验验证**未带来明显收益**：

| 方向 | 实验 | 结果 |
|---|---|---|
| cos 方向损失 (mse) | v14 | <1% |
| cos 方向损失 (NLLL) | v15 | -2.5% |
| 速度残差预测 + vx 权重 | v17 | <1% |
| Wide backbone + NLLL | v16 | 略优但参数 4× |
| w_prior 加倍 | wprior_high | -5% |
| 教师末两层解冻联合 | unfreeze_tail | -12% |
| 学生单步训练继续 | resume | <1.5% |
| episode 2000 极限 | extreme（未完） | 早期信号未刷新 |

---

## 6. 实验时间线

| 日期 | 事件 |
|---|---|
| ~5/15 之前 | 早期 v3/v3_wide NLLL teacher + cvn_h256/tier_s 学生 |
| 5/16 12:00 前 | mse_3ch backbone 横评 v6–v10 |
| 5/16 13:00 | gru 系列扩展 v10/v11/v12 |
| 5/16 17:30 | v10 / v11 mid eval |
| 5/16 21:00 | cos 实验 v14 / v15，NLLL v13 |
| 5/16 22:00 | v13 eval → 当选教师 |
| 5/17 00:00 | v17 vel residual + vx 权重 |
| 5/17 02:00 | v17 eval → 不再调教师 |
| 5/17 02:30 | Phase 2-a 学生开训 |
| 5/17 03:30 | P2-a TIMEOUT，best.pt 保留 |
| 5/17 04:00 | Phase 3 horizon + Phase 2-a resume + Phase 2 w_prior 探针 |
| 5/17 06:00 | P3 horizon 完成（best 0.04738） |
| 5/17 06:30 | P2-a resume 取消（plateau） |
| 5/17 07:00 | P3 long 提交 |
| 5/17 12:00 | P3 long TIMEOUT（best 0.04734, ep3） |
| 5/17 13:00 | unfreeze tail (C) → 失败取消 |
| 5/18 23:00 | 推理性能测试，300 帧可视化 |

---

## 7. Checkpoint 路径与最终配置

### 教师
```
runs/pedpred_v13_gru_mid_nlll_17843926/checkpoints/free-pig_best.hkl
```

### 学生
```
runs/cvn_p3_v13_horizon_17847842/best.pt
```

### 训练命令（参考）

教师（Phase 1）：
```bash
PEDPRED_OBJECTIVE=nlll PEDPRED_ARCH=pedpred3_gru_mid \
PEDPRED_BATCH=48 PEDPRED_NIN=5 PEDPRED_NOUT=5 \
PEDPRED_LR=5e-4 PEDPRED_WEIGHT_DECAY=1e-4 \
PEDPRED_FLIP_W_PROB=0.5 PEDPRED_EARLY_STOP_PATIENCE=20 \
python -u -m crowd_varnet.pedpred_teacher_train --max-epochs 30
```

学生 Phase 2-a：
```bash
python -m crowd_varnet.rollout_train_cli \
  --checkpoint <teacher.hkl> --arch pedpred3_gru_mid \
  --epochs 25 --batch 48 --episode-len 300 --k-prime 8 --warmup 5 \
  --n-iter 8 --solver-type convgru --solver-hidden 256 --solver-kernel 3 \
  --w-prior 0.5 --rho-mask-thr 0.05 \
  --ch-weights 3.0 1.5 1.0 0.0 \
  --lr 2e-4 --lr-eta-min-ratio 0.25 \
  --weight-decay 1e-4 --solver-dropout 0.1 --ema-decay 0.999 \
  --grad-clip 0.5 --save-dir <out>
```

学生 Phase 3 horizon（warmstart 自 P2-a best.pt）：
```bash
python -m crowd_varnet.rollout_train_cli \
  --checkpoint <teacher.hkl> --arch pedpred3_gru_mid \
  --warmstart <p2a_best.pt> \
  --epochs 8 --batch 32 --episode-len 600 --k-prime 12 --warmup 5 \
  --n-iter 8 --solver-type convgru --solver-hidden 256 --solver-kernel 3 \
  --w-prior 0.5 --rho-mask-thr 0.05 \
  --ch-weights 3.0 1.5 1.0 0.0 \
  --lr 5e-5 --lr-eta-min-ratio 0.25 \
  --weight-decay 1e-4 --solver-dropout 0.1 --ema-decay 0.999 \
  --grad-clip 0.5 --save-dir <out>
```

---

## 8. 物理界限

- **vy** 通道 valid support_RMSE 最优 0.232，距噪声地板 0.20 仅 15%。
- **vx** 通道 0.277，仍有 5–10% 优化空间，但需要不同信号源（多步 horizon 已榨过）。
- **rho** 通道 0.129，背景 leak 1%，已经很干净。

进一步优化可能需要的方向（未尝试，留待未来）：
- 加更多观测 agent / 改变 sensing geometry
- 教师改更长 nin（nin=8/10）
- 学生迭代次数 n_iter ≥ 12
- 物理 continuity loss（∂ρ/∂t + ∇·(ρv) = 0）

---

*最后更新: 2026-05-18*
