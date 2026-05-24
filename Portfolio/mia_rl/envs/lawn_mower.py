from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mia_rl.mdps.base import TabularMDP
from mia_rl.envs.gridworld import (ACTIONS, ACTION_TO_DELTA)

# =========================================================
# MAPA EM L
# =========================================================

L_MAP = np.array([
    [2,1,1,1,1,1,1,1,1],
    [1,0,1,0,1,0,1,0,1],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,1],
    [1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,1,1,3],
])

# =========================================================
# ENVIRONMENT
# =========================================================

@dataclass
class LawnMowerEnv(TabularMDP):

    grid: np.ndarray

    # =====================================================
    # INIT
    # =====================================================

    def __post_init__(self):

        self.valid_cells = set()

        self.start = None
        self.goal = None

        rows, cols = self.grid.shape

        self.n_rows = rows
        self.n_cols = cols

        for r in range(rows):
            for c in range(cols):

                value = self.grid[r, c]

                if value != 0:
                    self.valid_cells.add((r, c))

                if value == 2:
                    self.start = (r, c)

                if value == 3:
                    self.goal = (r, c)

        self.n_valid_cells = len(self.valid_cells)

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        visited = frozenset([self.start])

        return (self.start, visited)

    # =====================================================
    # STATES
    # =====================================================

    def states(self):

        return list(self.valid_cells)

    # =====================================================
    # TERMINAL
    # =====================================================

    def is_terminal(self, state):

        position, visited = state

        return (
            position == self.goal
            and
            len(visited) == self.n_valid_cells
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(self, state, action):

        position, visited = state

        # Estado terminal
        if self.is_terminal(state):
            return state, 0.0, True

        dr, dc = ACTION_TO_DELTA[action]

        r, c = position

        nr, nc = r + dr, c + dc

        next_position = (nr, nc)

        # =================================================
        # MOVIMENTO INVÁLIDO
        # =================================================

        if next_position not in self.valid_cells:

            return state, -5.0, False

        # =================================================
        # NOVA CÉLULA
        # =================================================

        new_visited = set(visited)

        if next_position in visited:

            reward = -2.0

        else:

            reward = +5.0

            new_visited.add(next_position)

        # Pequena penalização por movimento
        reward -= 0.1

        new_state = (
            next_position,
            frozenset(new_visited)
        )

        done = self.is_terminal(new_state)

        # =================================================
        # BONUS TERMINAL
        # =================================================

        if done:
            reward += 100.0

        return new_state, reward, done

    # =====================================================
    # ACTIONS
    # =====================================================

    def possible_actions(self, state):

        if self.is_terminal(state):
            return []

        return ACTIONS

    # =====================================================
    # TRANSITIONS
    # =====================================================

    def transitions(self, state, action):

        ns, r, done = self.step(state, action)

        return [(1.0, ns, r, done)]