from __future__ import annotations

from pathlib import Path
import sys

from mia_rl.envs.carrental import CarRentalMDP, CarRentalParams
from mia_rl.agents.planning.carrental import policy_iteration, value_iteration
from mia_rl.plots.carrental import plot_policy, plot_values

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run_carrental():

    gamma=0.9
    
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "carrental"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = CarRentalParams()
    mdp = CarRentalMDP(params)

    # Policy Iteration
    V_pi, policy_pi, history = policy_iteration(mdp, gamma=gamma)

    plot_policy(output_dir, "policy_iteration_policy.png", mdp, policy_pi, title="Policy Iteration")
    plot_values(output_dir, "policy_iteration_values.png", mdp, V_pi, title="Values (PI)")
    
    # Value Iteration
    V_vi, policy_vi, iters_vi = value_iteration(mdp, gamma=gamma)

    plot_policy(output_dir, "value_iteration_policy.png", mdp, policy_vi, title="Value Iteration")
    plot_values(output_dir, "value_iteration_policy.png", mdp, V_vi, title="Values (VI)")

    input("Press Enter to close...")
