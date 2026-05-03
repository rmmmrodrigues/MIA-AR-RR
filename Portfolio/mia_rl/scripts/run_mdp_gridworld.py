from __future__ import annotations

import sys
from pathlib import Path

from click import Tuple
import matplotlib.pyplot as plt
import numpy as np
from regex import V0

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#from mia_rl.agents.planning.dynamic_programming import (
from mia_rl.envs.gridworld import (
    policy_evaluation,
    policy_evaluation_Q,
    policy_evaluation_with_history,
    value_iteration,
    value_iteration_stochastic,
    zeros_V,
)
#from mia_rl.mdps.gridworld import ACTIONS, Gridworld, TrapGridworld
from mia_rl.envs.gridworld import ACTIONS, Gridworld, TrapGridworld

from mia_rl.plots.mdp_gridworld import plot_grid_values_and_policy
from mia_rl.policies.gridworld import greedy_policy_from_V, uniform_random_policy


def save_plot(fig, output_dir: Path, filename: str) -> None:
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)

    output_dir = PACKAGE_ROOT / "outputs" / "mdp_gridworld"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Gridworld()

    # From a non-terminal, moving into a wall keeps you in place but still costs -1
    print("From (0,1) with U:", env.step((0, 1), "U"))
    # From a non-terminal, moving toward terminal
    print("From (0,1) with L:", env.step((0, 1), "L"))
    # From terminal, stay terminal
    print("From (0,0) with R:", env.step((0, 0), "R"))

    V0 = zeros_V(env)
    V0

    policy = uniform_random_policy(env)
    policy[(1,1)]    


    gamma = 0.9
    V_pi, iters = policy_evaluation(env, policy, gamma=gamma, theta=1e-8)
    print("Converged in iterations:", iters)

    V_pi_hist, it_pi_hist = policy_evaluation_with_history(env, policy, gamma=gamma, theta=1e-8)

    # For policy evaluation, the policy is fixed (uniform random), so we don't need to derive it from V.
    # However, the plot_grid_values_and_policy expects a single action per state for arrows.
    # For a uniform random policy, we can choose not to show arrows or represent them as '·'.
    # Let's create a placeholder policy that always shows '·' for non-terminal states.
    plot_policy_for_pe = {
        s: '·' if env.is_terminal(s) else '?' for s in env.states()
    }

    # User-requested specific iterations to plot
    specific_iterations = [0, 1, 2, 3, 4, 8, 50, 100]

    # Filter out iterations that are beyond the actual history length
    indices_to_plot = sorted(list(set([it for it in specific_iterations if it < len(V_pi_hist)])))

    for i in indices_to_plot:
        current_V_pi = V_pi_hist[i]
        current_it_pi = it_pi_hist[i]
        fig, _ = plot_grid_values_and_policy(
            env, current_V_pi, plot_policy_for_pe,
            title=f"Policy Evaluation: V^π (uniform random π) (Iteration {current_it_pi})"
        )
        save_plot(fig, output_dir, f"plot_grid_values_and_policy_{current_it_pi}.png")

    V_star, iters_vi = value_iteration(env, gamma=gamma, theta=1e-8)
    print("Converged in iterations:", iters_vi)
    V_star

    pi_star = greedy_policy_from_V(env, V_star, gamma=gamma)
    pi_star[(2,2)], pi_star[(0,1)], pi_star[(1,0)]    

    print("Optimal policy:")
    print(pi_star)

    Q_pi, itq = policy_evaluation_Q(env, policy, gamma=gamma, theta=1e-8)
    print("Q^pi converged in iterations:", itq)

    # Verify V^pi(s) = sum_a pi(a|s) Q^pi(s,a)
    V_from_Q = zeros_V(env)
    for (r,c) in env.states():
        s = (r,c)
        if env.is_terminal(s):
            V_from_Q[r,c] = 0.0
        else:
            V_from_Q[r,c] = sum(policy[s][a]*Q_pi[r,c,a_index] for a_index,a in enumerate(ACTIONS))

    print("max |V_pi - V_from_Q| =", np.max(np.abs(V_pi - V_from_Q)))    

    fig, _ = plot_grid_values_and_policy(env, V_pi, None, title="Policy evaluation: V^π (uniform random π)")
    save_plot(fig, output_dir, f"plot_grid_values_and_policy_{current_it_pi}.png")

    fig, _ = plot_grid_values_and_policy(env, V_star, pi_star, title="Value iteration: V* and greedy policy")
    save_plot(fig, output_dir, "plot_grid_values_and_policy_value_iteration.png")


    # Exercise A: compare gammas
    for g in [0.5, 0.9, 0.99]:
        Vg, itg = value_iteration(env, gamma=g, theta=1e-8)
        pig = greedy_policy_from_V(env, Vg, gamma=g)
        print(f"\nGamma = {g} (value iteration iters={itg})")
        fig, _ = plot_grid_values_and_policy(env, Vg, None, title=f"V* and greedy policy (gamma={g})")
        save_plot(fig, output_dir, f"plot_grid_values_and_policy_{str(g).replace('.', '_')}.png")
        fig, _ = plot_grid_values_and_policy(env, Vg, pig, title=f"V* and greedy policy (gamma={g})")
        save_plot(fig, output_dir, f"plot_grid_values_and_policy_value_iteration_{str(g).replace('.', '_')}_policy.png")

    # Exercise B: trap cell with -10 on entry
    env_trap = TrapGridworld()
    V_trap, it_trap = value_iteration(env_trap, gamma=0.90, theta=1e-8)
    pi_trap = greedy_policy_from_V(env_trap, V_trap, gamma=0.90)
    print("Trap value iteration iters:", it_trap)
    fig, _ = plot_grid_values_and_policy(env_trap, V_trap, None, title="V* with trap at (0,2) reward -10")
    save_plot(fig, output_dir, "plot_grid_values_and_policy_trap_gridworld.png")
    fig, _ = plot_grid_values_and_policy(env_trap, V_trap, pi_trap, title="V* with trap at (0,2) reward -10")
    save_plot(fig, output_dir, "plot_grid_values_and_policy_trap_gridworld_policy.png")

    # Exercise C: Make the environment stochastic
    V_stoch, its = value_iteration_stochastic(env, gamma=0.9, theta=1e-8)
    pi_stoch = greedy_policy_from_V(env, V_stoch, gamma=0.9)
    print("Stochastic value iteration iters:", its)
    fig, _ = plot_grid_values_and_policy(env, V_stoch, None, title="Stochastic: V^pi (uniform)")
    save_plot(fig, output_dir, "plot_grid_values_and_policy_stochastic.png")
    fig, _ = plot_grid_values_and_policy(env, V_stoch, pi_stoch, title="V* with stochastic slip (0.8/0.1/0.1)")
    save_plot(fig, output_dir, "plot_grid_values_and_policy_stochastic_policy.png")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
