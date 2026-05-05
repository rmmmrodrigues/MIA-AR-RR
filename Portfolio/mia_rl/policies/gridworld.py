from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np

from mia_rl.envs.gridworld import ACTIONS, Gridworld, GridworldAction, GridworldState


# ======================================
# HELPERS
# ======================================

def greedy_action_from_V(env: Gridworld, V: np.ndarray, s: Tuple[int,int], gamma: float) -> str:
    """Helper: return argmax_a [r + gamma V(s')]."""
    best_a = None
    best_q = -np.inf
    for a in ACTIONS:
        ns, r, done = env.step(s, a)
        q = r + gamma * V[ns[0], ns[1]]
        if q > best_q:
            best_q = q
            best_a = a
    return best_a

# ======================================
# POLICIES
# ======================================

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


def policy_improvement(
    env: Gridworld,
    V: np.ndarray,
    old_policy_actions: Optional[Dict[Tuple[int,int], str]] = None,
    gamma: float = 0.9
) -> Tuple[Dict[Tuple[int,int], str], bool]:
    """Greedify policy w.r.t. V. Returns (new_policy_actions, stable)."""
    new_policy_actions: Dict[Tuple[int,int], str] = {}
    stable = True

    for s in env.states():
        if env.is_terminal(s):
            new_policy_actions[s] = "·"
            continue

        # TODO 1:
        # 1) compute the best action using one-step lookahead:
        #       q(s,a) = r(s,a) + gamma * V(s')
        # 2) pick the action with the highest q
        # 3) set stable=False when it changes

        # --- SOLUTION  ---
        best_a = greedy_action_from_V(env, V, s, gamma)
        new_policy_actions[s] = best_a

        if old_policy_actions is not None:
            if old_policy_actions.get(s, None) != best_a:
                stable = False
        else:
            stable = False  # if no reference policy provided, we can't declare stability
        # -----------------

    return new_policy_actions, stable