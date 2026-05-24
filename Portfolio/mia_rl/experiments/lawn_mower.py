from __future__ import annotations

from pathlib import Path
import sys

from mia_rl.envs.lawn_mower import LawnMowerEnv, L_MAP
from mia_rl.agents.control.q_learning import QLearningAgent
from mia_rl.plots.lawn_mower import plot_coverage_heatmap, plot_episode_lengths, plot_training_rewards


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

algorithm_name = "Q-Learning"
# =========================================================
# RUN EXPERIMENT
# =========================================================

def run_lawn_mower():

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "lawn_mower"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # ENVIRONMENT
    # =====================================================

    env = LawnMowerEnv(L_MAP)

    # =====================================================
    # AGENT
    # =====================================================

    agent = QLearningAgent(
        env,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    )

    # =====================================================
    # TRAIN
    # =====================================================

    rewards_history = agent.train(
        episodes=2000,
        max_steps=500,
    )

    # =====================================================
    # TRAINING REWARD CURVE
    # =====================================================
    plot_training_rewards(rewards_history, output_dir, algorithm_name)

    # =====================================================
    # EPISODE LENGTH CURVE
    # =====================================================
    plot_episode_lengths(rewards_history, output_dir, algorithm_name)    

    # =====================================================
    # EXECUTE LEARNED POLICY
    # =====================================================

    state = env.reset()

    trajectory = []

    visit_counts = {}

    done = False

    eval_steps = 0

    max_eval_steps = 500

    while not done and eval_steps < max_eval_steps:

        eval_steps += 1

        position, visited = state

        trajectory.append(position)

        # =================================================
        # VISIT COUNTS
        # =================================================

        if position not in visit_counts:
            visit_counts[position] = 0

        visit_counts[position] += 1

        # =================================================
        # GREEDY ACTION
        # =================================================

        action = agent.greedy_action(state)

        next_state, reward, done = env.step(
            state,
            action
        )

        state = next_state

    # =====================================================
    # FINAL POSITION
    # =====================================================

    position, visited = state

    trajectory.append(position)

    if position not in visit_counts:
        visit_counts[position] = 0

    visit_counts[position] += 1

    # =====================================================
    # METRICS
    # =====================================================

    coverage_ratio = len(visited) / env.n_valid_cells

    total_steps = len(trajectory)

    repeated_visits = (
        sum(visit_counts.values())
        -
        len(visit_counts)
    )

    efficiency = (
        len(visit_counts)
        /
        total_steps
    )

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print()
    print("======================================")
    print("COVERAGE RESULTS")
    print("======================================")
    print(f"Coverage Ratio : {coverage_ratio:.2%}")
    print(f"Total Steps    : {total_steps}")
    print(f"Repeated Visits: {repeated_visits}")
    print(f"Efficiency     : {efficiency:.2%}")
    print("======================================")

    # =====================================================
    # FINAL PLOT
    # =====================================================
    plot_coverage_heatmap(env, visit_counts, trajectory, position, output_dir, algorithm_name)

    input("Press Enter to close...")
