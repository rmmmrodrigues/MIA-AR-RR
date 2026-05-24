from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from mia_rl.envs.carrental import CarRentalMDP

def q_from_v(mdp: CarRentalMDP, V: np.ndarray, s: Tuple[int,int], a: int, gamma: float) -> float:
    """Compute q(s,a) = E[ reward + gamma * V(s') ] under the model.
    “If I choose this action tonight, what is my total expected profit — now and in the long run?”
    """
    # TODO (A): implement expected return.
    # Steps:
    # 1) Get p1(n1'), p2(n2') and expected revenue from mdp.expected_transition(s,a)
    # 2) Expected next value = sum_{n1',n2'} p1[n1']*p2[n2']*V[n1',n2']
    #   a) we need to check possible outcumes for one location, and on each outcome, check all outcomes for the other location
    # 3) Moving cost = cost_per_moved * |a|
    # 4) q = reward + gamma * expected_next_value
    #
    # --- SOLUTION ---
    p_next_1, p_next_2, exp_revenue = mdp.expected_transition(s, a)

    # expected next value via outer product
    exp_next = 0.0
    for n1p, p1v in enumerate(p_next_1):    # try all possible outcomes for location 1
        if p1v == 0.0:
            continue
        for n2p, p2v in enumerate(p_next_2):    # for each of those, try all possible outcomes for location 2
            if p2v == 0.0:
                continue
            exp_next += p1v * p2v * V[n1p, n2p]   #Probability of n1p cars at loc1 * Probability of n2p cars at loc2 * V[next_state]


    move_cost = mdp.params.cost_per_moved * abs(a)
    reward = exp_revenue - move_cost

    return reward + gamma * exp_next

def bellman_expectation_backup_v(
    mdp: CarRentalMDP,
    V: np.ndarray,
    s: Tuple[int, int],
    policy: Dict[Tuple[int, int], int],
    gamma: float,
) -> float:
    """(T^π V)(s) for a deterministic policy π(s)."""
    a = policy[s]
    return q_from_v(mdp, V, s, a, gamma)


def bellman_optimality_backup_v(
    mdp: CarRentalMDP,
    V: np.ndarray,
    s: Tuple[int, int],
    gamma: float,
) -> float:
    """(T* V)(s) = max_a q(s,a)"""
    best = -float("inf")
    for a in mdp.possible_actions(s):
        best = max(best, q_from_v(mdp, V, s, a, gamma))
    return best

def zeros_V(mdp: CarRentalMDP) -> np.ndarray:
    return np.zeros((mdp.params.max_cars_1 + 1, mdp.params.max_cars_2 + 1), dtype=float)
#----------------------------------
# algoritmos que PRODUZEM policies
#----------------------------------
def policy_evaluation(
    mdp: CarRentalMDP,
    policy: Dict[Tuple[int,int], int],
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    V = zeros_V(mdp)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()

        # TODO (B): sweep all states and update V with the current policy action.
        # Use: delta = max(delta, |V_new - V_old|). Stop if delta < theta.
        #
        # --- SOLUTION ---
        for s in mdp.states():
            v_new = bellman_expectation_backup_v(mdp, V_old, s, policy, gamma)
            delta = max(delta, abs(v_new - V[s[0], s[1]]))
            V[s[0], s[1]] = v_new

        if delta < theta:
            return V, it + 1
        # ----------------

    return V, max_iters

def policy_improvement(
    mdp: CarRentalMDP,
    V: np.ndarray,
    old_policy: Optional[Dict[Tuple[int,int], int]],
    gamma: float,
) -> Tuple[Dict[Tuple[int,int], int], bool]:
    new_policy: Dict[Tuple[int,int], int] = {}
    stable = True

    for s in mdp.states():
        # TODO (C): choose the greedy action using q_from_v().
        # 1) loop actions
        # 2) pick argmax q(s,a)
        # 3) compare to old_policy[s] (if provided) to set stable=False if changed
        #
        # --- SOLUTION ---
        best_a = None
        best_q = -np.inf
        for a in mdp.possible_actions(s):
            q = q_from_v(mdp, V, s, a, gamma)
            if q > best_q:
                best_q = q
                best_a = a

        new_policy[s] = best_a

        if old_policy is not None and old_policy[s] != best_a:
            stable = False
        if old_policy is None:
            stable = False

    return new_policy, stable

def policy_iteration(
    mdp: CarRentalMDP,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_outer: int = 50,
):
    # Initialize with a simple policy: always move 0 cars
    policy = {s: 0 for s in mdp.states()}
    history = []

    for outer in range(max_outer):
        # TODO (D): Implement policy iteration loop.
        #
        # --- SOLUTION ---
        V, iters = policy_evaluation(mdp, policy, gamma=gamma, theta=theta)
        new_policy, stable = policy_improvement(mdp, V, old_policy=policy, gamma=gamma)

        history.append((outer, iters, V.copy(), new_policy.copy()))


        policy = new_policy
        if stable:
            break

    return V, policy, history

def value_iteration(
    mdp: CarRentalMDP,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iters: int = 10_000,
):
    V = zeros_V(mdp)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()

        # TODO (E): Bellman optimality backup for each state.
        # v_new = ...
        #
        # --- SOLUTION ---
        for s in mdp.states():
            v_new = bellman_optimality_backup_v(mdp, V_old, s, gamma)
            delta = max(delta, abs(v_new - V[s[0], s[1]]))
            V[s[0], s[1]] = v_new

        if delta < theta:
            break

    # derive greedy policy from the FINAL V (use V, not V_old)
    pi = {}
    for s in mdp.states():
        best_a = None
        best_q = -np.inf
        for a in mdp.possible_actions(s):
            q = q_from_v(mdp, V, s, a, gamma)
            if q > best_q:
                best_q = q
                best_a = a
        pi[s] = best_a

    return V, pi, it + 1
