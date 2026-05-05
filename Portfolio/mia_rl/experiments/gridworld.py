from __future__ import annotations

from email import policy
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from mia_rl.envs.gridworld import Gridworld
from mia_rl.agents.planning.gridworld import (
    policy_evaluation,
    policy_evaluation_with_history,
    value_iteration,
    value_iteration_stochastic,
    policy_iteration
)
from mia_rl.policies.gridworld import (
    uniform_random_policy,
    greedy_policy_from_V,
    policy_improvement,
)
from mia_rl.plots.mdp_gridworld import plot_grid, plot_grid_values_and_policy


def run_gridworld(output_dir: Path, gamma: float = 0.9) -> None:
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

    fig, _ = plot_grid(env, V_pi0, policy=None, title="Policy evaluation: V^π (uniform random π)")
    fig.savefig(output_dir / "policy_evaluation.png")


    # ======================================
    # POLICY IMPROVEMENT
    # ======================================
    pi1_actions, _ = policy_improvement(env, V_pi0, old_policy_actions=None, gamma=gamma)
    fig, _ = plot_grid(env, V_pi0, policy=pi1_actions, title="Greedy policy w.r.t. V^π (arrows)")
    fig.savefig(output_dir / "policy_improvement.png")

    # ======================================
    # POLICY ITERATION
    # ======================================
    V_star, pi_star_actions, hist = policy_iteration(env, gamma=gamma)
    print("Policy iteration outer loops:", len(hist))
    V_star

    fig, _ = plot_grid(env, V_star, policy=pi_star_actions, title="Policy Iteration: V* and π* (greedy actions)")
    fig.savefig(output_dir / "policy_iteration.png")

    pi_star_actions


    num_plots = len(hist)
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 6, 6)) # Adjust figsize for horizontal plots

    # Ensure axes is an array even for a single plot
    if num_plots == 1:
        axes = [axes]

    for i, (outer_iter, pe_iters, V_hist, pi_actions_hist) in enumerate(hist):
        ax = axes[i]
        fig, _ = plot_grid(env, V_hist, policy=pi_actions_hist, title=f"Policy Iteration (Outer Loop) {outer_iter}\nPolicy Evaluation (Inner Loop) Itrs: {pe_iters}", ax=ax)
        
    fig.savefig(output_dir / "policy_iteration_all.png")
    plt.show()


    # V_pi, iters = policy_evaluation(env, policy, gamma=gamma, theta=1e-8)
    # print("Policy evaluation iterations:", iters)

    # V_hist, it_hist = policy_evaluation_with_history(env, policy, gamma=gamma, theta=1e-8)

    # # plot intermediate iterations
    # selected_iters = [0, 1, 2, 3, 4, 8, 50, 100]
    # selected_iters = [i for i in selected_iters if i < len(V_hist)]

    # dummy_policy = {
    #     s: "·" if env.is_terminal(s) else "?"
    #     for s in env.states()
    # }

    # for i in selected_iters:
    #     fig, _ = plot_grid_values_and_policy(
    #         env,
    #         V_hist[i],
    #         dummy_policy,
    #         title=f"Policy Evaluation (iter {it_hist[i]})"
    #     )
    #     fig.savefig(output_dir / f"policy_eval_{it_hist[i]:03d}.png")

    # # ======================================
    # # POLICY IMPROVEMENT
    # # ======================================
    # pi_greedy, _ = policy_improvement(env, V_pi, gamma=gamma)

    # fig, _ = plot_grid_values_and_policy(
    #     env,
    #     V_pi,
    #     pi_greedy,
    #     title="Greedy policy from V^pi"
    # )
    # fig.savefig(output_dir / "policy_improvement.png")

    # # ======================================
    # # VALUE ITERATION
    # # ======================================
    # V_star, it_vi = value_iteration(env, gamma=gamma)
    # print("Value iteration iterations:", it_vi)

    # pi_star = greedy_policy_from_V(env, V_star, gamma=gamma)

    # fig, _ = plot_grid_values_and_policy(
    #     env,
    #     V_star,
    #     pi_star,
    #     title="Optimal V* and policy"
    # )
    # fig.savefig(output_dir / "value_iteration.png")

    # # ======================================
    # # STOCHASTIC ENV
    # # ======================================
    # V_stoch, it_stoch = value_iteration_stochastic(env, gamma=gamma)
    # print("Stochastic VI iterations:", it_stoch)

    # pi_stoch = greedy_policy_from_V(env, V_stoch, gamma=gamma)

    # fig, _ = plot_grid_values_and_policy(
    #     env,
    #     V_stoch,
    #     pi_stoch,
    #     title="Stochastic optimal policy"
    # )
    # fig.savefig(output_dir / "value_iteration_stochastic.png")

    print(f"Saved results to {output_dir}")