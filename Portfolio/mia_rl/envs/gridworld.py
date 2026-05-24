from __future__ import annotations

from typing import List, Literal, Tuple
from dataclasses import dataclass

from mia_rl.mdps.base import TabularMDP

# ======================================
# CONSTANTS
# ======================================

LEFT_OF = {"U":"L","D":"R","L":"D","R":"U"}
RIGHT_OF = {"U":"R","D":"L","L":"U","R":"D"}

ACTIONS = ["U", "D", "L", "R"]

GridworldAction = Literal["U", "D", "L", "R"]
GridworldState = Tuple[int, int]

ACTION_TO_DELTA = {
    "U": (-1, 0),
    "D": ( 1, 0),
    "L": ( 0,-1),
    "R": ( 0, 1),
}

# ======================================
# ENVIRONMENT
# ======================================

@dataclass(frozen=True)
class Gridworld(TabularMDP[Tuple[int,int], str]):
    n_rows: int = 4
    n_cols: int = 4
    terminal_states: Tuple[Tuple[int,int], ...] = ((0,0), (3,3))
    step_reward: float = -1.0

    def states(self) -> List[Tuple[int,int]]:
        return [(row,column) for row in range(self.n_rows) for column in range(self.n_cols)]

    def is_terminal(self, state: Tuple[int,int]) -> bool:
        return state in self.terminal_states

    def step(self, state: Tuple[int,int], action: str) -> Tuple[Tuple[int,int], float, bool]:
        # Se já está em estado terminal
        if self.is_terminal(state):
            return state, 0.0, True

        # TODO 1--- YOUR CODE STARTS HERE ---
        # Movimento proposto
        dr, dc = ACTION_TO_DELTA[action]
        r, c = state
        print(f"State: {state}, Action: {action}, Delta: {(dr, dc)}")
        next_r, next_c = r + dr, c + dc

        # Verifica limites do grid
        if 0 <= next_r < self.n_rows and 0 <= next_c < self.n_cols:
            next_state = (next_r, next_c)
        else:
            next_state = state  # bateu na parede

        reward = self.step_reward
        done_flag = self.is_terminal(next_state)
        # --- YOUR CODE ENDS HERE ---
        
        return next_state, reward, done_flag
    
    def possible_actions(self, state):
        if self.is_terminal(state):
            return []
        return ACTIONS

    def transitions(self, state, action):
        ns, r, done = self.step(state, action)
        return [(1.0, ns, r, done)]

class TrapGridworld(Gridworld):
    def __init__(self, trap=(0, 2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "trap", trap)

    def step(self, s: Tuple[int,int], a: str):
        ns, r, done = super().step(s, a)

        if ns == self.trap:
            r = -10.0

        return ns, r, done