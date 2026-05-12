"""Partial-observation geometry and multi-agent motion (from ENKF_pedpred, trimmed)."""
from __future__ import annotations

import numpy as np


class GeneratePartialObs:
    def __init__(self, GRID_SIZE, STATE_SHAPE, agent_pos, sensing_range):
        self.GRID_SIZE = GRID_SIZE
        self.STATE_SHAPE = STATE_SHAPE
        self.agent_pos = agent_pos
        self.sensing_range = sensing_range

        self.num_features = STATE_SHAPE[0]
        self.total_cells = GRID_SIZE[0] * GRID_SIZE[1]
        self.state_dim = int(np.prod(STATE_SHAPE))

    def cells_within_range(self):
        rows, cols = self.GRID_SIZE
        cells = []
        r0, c0 = self.agent_pos
        for r in range(rows):
            for c in range(cols):
                if np.sqrt((r - r0) ** 2 + (c - c0) ** 2) <= self.sensing_range:
                    cells.append((r, c))
        return cells

    def get_observation_matrix(self):
        observed_cells = self.cells_within_range()
        m = self.num_features * len(observed_cells)
        C = np.zeros((m, self.state_dim))
        for i, (r, c) in enumerate(observed_cells):
            for f in range(self.num_features):
                row_idx = i * self.num_features + f
                col_idx = f * self.total_cells + r * self.GRID_SIZE[1] + c
                C[row_idx, col_idx] = 1.0
        return C, observed_cells

    def reconstruct_observation(self, obs_vector, C, fill_value=np.nan):
        reconstructed = np.full(self.STATE_SHAPE, fill_value, dtype=float)
        obs_indices = C.nonzero()[1]
        for row_idx, state_idx in enumerate(obs_indices):
            f = state_idx // self.total_cells
            rc = state_idx % self.total_cells
            r, c = divmod(rc, self.GRID_SIZE[1])
            reconstructed[f, r, c] = obs_vector[row_idx]
        return reconstructed


class MultiAgent:
    def __init__(self, grid_size, state_shape, sensing_range, num_agents=2, rng=None):
        self.grid_size = grid_size
        self.state_shape = state_shape
        self.sensing_range = sensing_range
        self.num_agents = num_agents
        self._rng = rng

        rnd = np.random if self._rng is None else self._rng
        self.positions = [
            (int(rnd.randint(0, grid_size[0])), int(rnd.randint(0, grid_size[1])))
            for _ in range(num_agents)
        ]
        self.goals = [
            (int(rnd.randint(0, grid_size[0])), int(rnd.randint(0, grid_size[1])))
            for _ in range(num_agents)
        ]

    def move_agents(self):
        new_positions = []
        for (r, c), (gr, gc) in zip(self.positions, self.goals):
            dr = np.sign(gr - r)
            dc = np.sign(gc - c)
            new_r = min(max(r + dr, 0), self.grid_size[0] - 1)
            new_c = min(max(c + dc, 0), self.grid_size[1] - 1)
            new_positions.append((new_r, new_c))
        self.positions = new_positions

    def get_joint_observation(self):
        all_cells = set()
        C_blocks = []
        for pos in self.positions:
            obs_gen = GeneratePartialObs(self.grid_size, self.state_shape, pos, self.sensing_range)
            C, cells = obs_gen.get_observation_matrix()
            all_cells.update(cells)
            C_blocks.append(C)
        C_joint = np.vstack(C_blocks)
        return C_joint, sorted(all_cells)
