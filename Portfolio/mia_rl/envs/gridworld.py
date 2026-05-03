from __future__ import annotations
from typing import List, Literal, Tuple

import numpy as np
from pyparsing import Dict

from dataclasses import dataclass

from mia_rl.mdps.base import TabularMDP

from mia_rl.mdps.base import TabularMDP

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

def zeros_V(env: Gridworld) -> np.ndarray:
    return np.zeros((env.n_rows, env.n_cols), dtype=float)


def zeros_Q(env: Gridworld) -> np.ndarray:
    # Q[r,c,a_index]
    return np.zeros((env.n_rows, env.n_cols, len(ACTIONS)), dtype=float)


def bellman_expectation_update(
    env: Gridworld, V: np.ndarray, policy: Dict[Tuple[int,int], Dict[str,float]],
    state: Tuple[int,int], gamma: float
) -> float:
    """Return the updated V(s) using the Bellman expectation backup."""
    if env.is_terminal(state):
        return 0.0

    # TODO 2: compute expected value over actions under pi(a|s)y alternating between evaluati
    # Hints:
    #  - loop over actions
    #  - env.step(s,a) gives (s', r, done)
    #  - use V[s'] (i.e., V[nr,nc])
    v_new = 0.0
    # --- YOUR CODE STARTS HERE ---
    for a, p in policy[state].items():
        ns, r, done = env.step(state, a)
        v_new += p * (r + gamma * V[ns[0], ns[1]])
    # --- YOUR CODE ENDS HERE ---
    return v_new


def policy_evaluation(
    env: Gridworld,
    policy: Dict[Tuple[int,int], Dict[str,float]],
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    """Iterative policy evaluation."""
    V = zeros_V(env)

    for it in range(max_iters):
        delta = 0.0

        # TODO 3: sweep all states and update V in-place (or into a copy)
        # Use delta = max(delta, |V_new - V_old|). Stop if delta < theta.
        # --- YOUR CODE STARTS HERE ---
        V_old = V.copy()
        for state in env.states():
            v_new = bellman_expectation_update(env, V_old, policy, state, gamma)
            delta = max(delta, abs(v_new - V[state[0], state[1]]))
            V[state[0], state[1]] = v_new
        # --- YOUR CODE ENDS HERE ---

        if delta < theta:
            return V, it + 1

    return V, max_iters

def policy_evaluation_with_history(
    env: Gridworld,
    policy: Dict[Tuple[int,int], Dict[str,float]],
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[List[np.ndarray], List[int]]:
    V = zeros_V(env)
    V_history = [V.copy()]
    iters_history = [0]

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for state in env.states():
            v_new = bellman_expectation_update(env, V_old, policy, state, gamma)
            delta = max(delta, abs(v_new - V[state[0], state[1]]))
            V[state[0], state[1]] = v_new
        V_history.append(V.copy())
        iters_history.append(it + 1)
        if delta < theta:
            return V_history, iters_history

    return V_history, max_iters

def bellman_optimality_update(env: Gridworld, V: np.ndarray, s: Tuple[int,int], gamma: float) -> float:
    if env.is_terminal(s):
        return 0.0
    best = -np.inf
    for a in ACTIONS:
        ns, r, done = env.step(s, a)
        best = max(best, r + gamma * V[ns[0], ns[1]])
    return best

def value_iteration(
    env: Gridworld,
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    V = zeros_V(env)
    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for s in env.states():
            v_new = bellman_optimality_update(env, V_old, s, gamma)
            delta = max(delta, abs(v_new - V[s[0], s[1]]))
            V[s[0], s[1]] = v_new
        if delta < theta:
            return V, it + 1
    return V, max_iters


def policy_evaluation_Q(
    env: Gridworld,
    pi: Dict[Tuple[int,int], Dict[str,float]],
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    Q = zeros_Q(env)

    for it in range(max_iters):
        delta = 0.0
        Q_old = Q.copy()

        for (r,c) in env.states():
            s = (r,c)
            if env.is_terminal(s):
                Q[r,c,:] = 0.0
                continue

            for a_index, a in enumerate(ACTIONS):
                ns, reward, done = env.step(s, a)
                nr, nc = ns

                # Expected over next action under pi
                exp_next = 0.0
                for aj, a2 in enumerate(ACTIONS):
                    exp_next += pi[ns][a2] * Q_old[nr, nc, aj]

                q_new = reward + gamma * exp_next
                delta = max(delta, abs(q_new - Q[r,c,a_index]))
                Q[r,c,a_index] = q_new

        if delta < theta:
            return Q, it + 1

    return Q, max_iters

def expected_backup_optimal_stochastic(env: Gridworld, V: np.ndarray, s: Tuple[int,int], a: str, gamma: float) -> float:
    if env.is_terminal(s):
        return 0.0
    outcomes = [(a, 0.8), (LEFT_OF[a], 0.1), (RIGHT_OF[a], 0.1)]
    exp = 0.0
    for a_eff, p in outcomes:
        ns, r, done = env.step(s, a_eff)
        exp += p * (r + gamma * V[ns[0], ns[1]])
    return exp

def value_iteration_stochastic(env: Gridworld, gamma: float, theta: float=1e-6, max_iters: int=10_000):
    V = zeros_V(env)
    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for s in env.states():
            if env.is_terminal(s):
                V[s[0], s[1]] = 0.0
                continue
            best = -np.inf
            for a in ACTIONS:
                best = max(best, expected_backup_optimal_stochastic(env, V_old, s, a, gamma))
            delta = max(delta, abs(best - V[s[0], s[1]]))
            V[s[0], s[1]] = best
        if delta < theta:
            return V, it+1
    return V, max_iters

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

# class TrapGridworld(Gridworld):
#     def __init__(self, trap=(0, 2), *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.trap = trap

#     def step(self, s: Tuple[int,int], a: str):
#         ns, r, done = super().step(s, a)

#         if ns == self.trap:
#             r = -10.0

#         return ns, r, done
class TrapGridworld(Gridworld):
    def __init__(self, trap=(0, 2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "trap", trap)

    def step(self, s: Tuple[int,int], a: str):
        ns, r, done = super().step(s, a)

        if ns == self.trap:
            r = -10.0

        return ns, r, done