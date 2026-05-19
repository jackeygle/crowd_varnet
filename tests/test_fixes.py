"""
针对三处 bug 修复的回归测试，以及核心模块的基本正确性检查。
不依赖真实数据，全部用随机 tensor。
"""
import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

B, T, H, W = 2, 5, 36, 12


class DummyPedPred(nn.Module):
    """最简先验：返回 history 最后一帧，形状 [B,1,4,H,W]。"""
    def forward(self, inp, hidden=None, *, horizon=1):
        t = inp if torch.is_tensor(inp) else torch.as_tensor(inp)
        if t.dim() == 5:
            t = t[:, -1]
        return t.unsqueeze(1)


def _make_model(**kwargs):
    from crowd_varnet.models.varnet import CrowdVarNet
    return CrowdVarNet(DummyPedPred(), freeze_phi=True, T_hist=T, n_iter=4, **kwargs)


def _make_batch():
    history  = torch.rand(B, T, 4, H, W)
    obs_mask = torch.zeros(B, 1, H, W)
    obs_mask[:, :, :, : W // 2] = 1.0
    obs      = torch.rand(B, 4, H, W) * obs_mask
    x_gt     = torch.rand(B, 4, H, W)
    return history, obs, obs_mask, x_gt


# ---------------------------------------------------------------------------
# Fix 1: adapter called only once in compute_loss
# ---------------------------------------------------------------------------

class _CountingPedPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inp, hidden=None, *, horizon=1):
        self.calls += 1
        t = inp if torch.is_tensor(inp) else torch.as_tensor(inp)
        if t.dim() == 5:
            t = t[:, -1]
        return t.unsqueeze(1)


def test_adapter_called_once_in_compute_loss():
    from crowd_varnet.models.varnet import CrowdVarNet
    counter = _CountingPedPred()
    model = CrowdVarNet(counter, freeze_phi=True, T_hist=T, n_iter=4)
    history, obs, obs_mask, x_gt = _make_batch()
    counter.calls = 0
    model.compute_loss(history, obs, obs_mask, x_gt)
    assert counter.calls == 1, f"adapter called {counter.calls} times, expected 1"


# ---------------------------------------------------------------------------
# Fix 2: InitGate receives gradients after a few training steps
# ---------------------------------------------------------------------------

def test_init_gate_receives_gradients():
    model = _make_model(init_gate=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, obs, obs_mask, x_gt = _make_batch()

    # 训练两步（第一步前层梯度为零是正常的零初始化行为）
    for _ in range(2):
        opt.zero_grad()
        loss, _ = model.compute_loss(history, obs, obs_mask, x_gt)
        loss.backward()
        opt.step()

    # 第二步后所有 InitGate 参数应有非 None 梯度
    for name, p in model.init_gate.named_parameters():
        assert p.grad is not None, f"InitGate param {name} has no gradient"


# ---------------------------------------------------------------------------
# Fix 3: sensor masks differ across sample indices
# ---------------------------------------------------------------------------

def test_sensor_mask_differs_across_idx():
    from crowd_varnet.deps.enkf_sensors import MultiAgent
    from crowd_varnet.datasets.sensors import spatial_sensor_mask

    sensing_range = 5.0
    num_agents = 3
    seed = 0
    G = 10

    masks = []
    for idx in range(3):
        rng = np.random.RandomState(seed + idx * 100003)
        agents = MultiAgent(
            (H, W), (4, H, W),
            sensing_range=sensing_range,
            num_agents=num_agents,
            rng=rng,
        )
        for _ in range(G + 1):
            agents.move_agents()
        masks.append(spatial_sensor_mask(H, W, list(agents.positions), sensing_range))

    assert not torch.equal(masks[0], masks[1]), "idx=0 and idx=1 produced identical masks"
    assert not torch.equal(masks[0], masks[2]), "idx=0 and idx=2 produced identical masks"


# ---------------------------------------------------------------------------
# Core: smoke — forward + compute_loss shapes and loss is finite
# ---------------------------------------------------------------------------

def test_forward_shape():
    model = _make_model()
    history, obs, obs_mask, _ = _make_batch()
    x_hat = model.forward(history, obs, obs_mask)
    assert x_hat.shape == (B, 4, H, W)


def test_compute_loss_finite():
    model = _make_model()
    history, obs, obs_mask, x_gt = _make_batch()
    loss, info = model.compute_loss(history, obs, obs_mask, x_gt)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    for k, v in info.items():
        assert np.isfinite(v), f"info[{k!r}] is not finite: {v}"


def test_compute_loss_with_init_gate():
    model = _make_model(init_gate=True)
    history, obs, obs_mask, x_gt = _make_batch()
    loss, _ = model.compute_loss(history, obs, obs_mask, x_gt)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("solver_type", ["scalar", "gru", "convgru"])
def test_solver_types(solver_type):
    model = _make_model(solver_type=solver_type)
    history, obs, obs_mask, x_gt = _make_batch()
    loss, _ = model.compute_loss(history, obs, obs_mask, x_gt)
    assert torch.isfinite(loss), f"solver_type={solver_type} produced non-finite loss"


# ---------------------------------------------------------------------------
# Core: VariationalCost
# ---------------------------------------------------------------------------

def test_variational_cost_finite():
    from crowd_varnet.models.cost import VariationalCost
    cost_fn = VariationalCost()
    x       = torch.rand(B, 4, H, W)
    obs     = torch.rand(B, 4, H, W)
    obs_mask= torch.zeros(B, 1, H, W); obs_mask[:, :, :, :W//2] = 1.0
    x_prior = torch.rand(B, 4, H, W)
    total, o, p = cost_fn(x, obs, obs_mask, x_prior)
    assert torch.isfinite(total)
    assert torch.isfinite(o)
    assert torch.isfinite(p)


# ---------------------------------------------------------------------------
# Core: clip_crowd_state respects bounds
# ---------------------------------------------------------------------------

def test_clip_crowd_state_bounds():
    from crowd_varnet.models.cost import clip_crowd_state
    x = torch.randn(B, 4, H, W) * 10
    out = clip_crowd_state(x)
    assert out[:, 0:1].min() >= 0.0 and out[:, 0:1].max() <= 5.0, "density out of [0,5]"
    assert out[:, 1:3].min() >= -5.0 and out[:, 1:3].max() <= 5.0, "velocity out of [-5,5]"
    assert out[:, 3:4].min() >= 0.0 and out[:, 3:4].max() <= 2.0, "var out of [0,2]"


# ---------------------------------------------------------------------------
# Core: spatial_sensor_mask geometry
# ---------------------------------------------------------------------------

def test_spatial_sensor_mask_center_visible():
    from crowd_varnet.datasets.sensors import spatial_sensor_mask
    r0, c0 = H // 2, W // 2
    mask = spatial_sensor_mask(H, W, [(r0, c0)], sensing_range=3.0)
    assert mask.shape == (1, H, W)
    assert mask[0, r0, c0] == 1.0, "center cell should be visible"


def test_spatial_sensor_mask_far_invisible():
    from crowd_varnet.datasets.sensors import spatial_sensor_mask
    mask = spatial_sensor_mask(H, W, [(0, 0)], sensing_range=2.0)
    assert mask[0, H - 1, W - 1] == 0.0, "far corner should be invisible"
