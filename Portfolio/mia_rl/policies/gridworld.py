from __future__ import annotations
from typing import Dict, Tuple

import numpy as np

from mia_rl.envs.gridworld import ACTIONS, Gridworld, GridworldAction, GridworldState


def uniform_random_policy(env: Gridworld) -> Dict[Tuple[int,int], Dict[str, float]]:
    policy = {}
    for s in env.states():
        if env.is_terminal(s):
            policy[s] = {a: 0.0 for a in ACTIONS}
        else:
            policy[s] = {a: 1.0/len(ACTIONS) for a in ACTIONS}
    return policy

def greedy_policy_from_V(env: Gridworld, V: np.ndarray, gamma: float) -> Dict[Tuple[int,int], str]:
    pi_greedy = {}
    for s in env.states():
        if env.is_terminal(s):
            pi_greedy[s] = "·"
            continue

        # TODO 4: get the best 'a' and 'q'
        # --- YOUR CODE STARTS HERE ---
        best_a = None
        best_q = -np.inf
        for a in ACTIONS:
            ns, r, done = env.step(s, a)
            q = r + gamma * V[ns[0], ns[1]]
            if q > best_q:
                best_q = q
                best_a = a
        pi_greedy[s] = best_a

        # --- YOUR CODE ENDS HERE ---
    return pi_greedy