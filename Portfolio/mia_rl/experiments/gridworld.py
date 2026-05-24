from __future__ import annotations

from pathlib import Path

import numpy as np

from mia_rl.plots.gridworld import plot_policy_evaluation, plot_policy_improvement, plot_policy_iteration, plot_policy_iteration_all
from mia_rl.envs.gridworld import Gridworld
from mia_rl.agents.planning.gridworld import (policy_evaluation, policy_iteration)
from mia_rl.policies.gridworld import (uniform_random_policy, policy_improvement)

def run_gridworld():

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "gridworld"
    output_dir.mkdir(parents=True, exist_ok=True)

    gamma: float = 0.9

    np.set_printoptions(precision=3, suppress=True)

    env = Gridworld()
    env.states()[:5], env.terminal_states

    # ======================================
    # POLICY
    # ======================================
    pi0 = uniform_random_policy(env)
    pi0[(1,1)]

    # ======================================
    # POLICY EVALUATION
    # ======================================
    gamma = 0.9
    V_pi0, iters = policy_evaluation(env, pi0, gamma=gamma, theta=1e-8)
    print("Policy evaluation converged in iterations:", iters)
    V_pi0

    plot_policy_evaluation(output_dir, env, V_pi0)

    # ======================================
    # POLICY IMPROVEMENT
    # ======================================
    pi1_actions, _ = policy_improvement(env, V_pi0, old_policy_actions=None, gamma=gamma)
    plot_policy_improvement(output_dir, env, V_pi0, pi1_actions)

    # ======================================
    # POLICY ITERATION
    # ======================================
    V_star, pi_star_actions, hist = policy_iteration(env, gamma=gamma)
    print("Policy iteration outer loops:", len(hist))
    V_star

    plot_policy_iteration(output_dir, env, V_star, pi_star_actions)

    pi_star_actions


    num_plots = len(hist)

    plot_policy_iteration_all(output_dir, num_plots, env, hist)

    input("Press Enter to close...")
