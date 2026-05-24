from __future__ import annotations

from pathlib import Path
import sys

from mia_rl.plots.lawn_mower import plot_coverage_heatmap, plot_episode_lengths, plot_training_rewards
from mia_rl.envs.lawn_mower import LawnMowerEnv, L_MAP
from mia_rl.agents.control.monte_carlo import MonteCarloControl
from mia_rl.core.base import Transition

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

algorithm_name = "Monte Carlo"
# =========================================================
# RUN MONTE CARLO EPISODE
# =========================================================
def run_monte_carlo_episode(
    env: LawnMowerEnv,
    agent: MonteCarloControl,
    max_steps: int = 500,
):

    state = env.reset()

    total_reward = 0.0

    steps = 0

    done = False

    eval_steps = 0

    max_eval_steps = 500

    while not done and eval_steps < max_eval_steps:

        eval_steps += 1

        position, visited = state

        trajectory = []

        visit_counts = {}

        trajectory.append(position)

        if position not in visit_counts:
            visit_counts[position] = 0

        visit_counts[position] += 1

        action = agent.greedy_action(state)

        next_state, reward, done = env.step(
            state,
            action,
        )

        state = next_state

        # =============================================
        # SELECT ACTION
        # =============================================

        action = agent.select_action(
            state
        )

        # =============================================
        # ENVIRONMENT STEP
        # =============================================

        next_state, reward, done = (
            env.step(
                state,
                action,
            )
        )

        total_reward += reward

        steps += 1

        # =============================================
        # STORE TRANSITION
        # =============================================

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        agent.update_transition(
            transition
        )

        # =============================================
        # MOVE TO NEXT STATE
        # =============================================

        state = next_state

    # =============================================
    # FORCE EPISODE FINALIZATION
    # =============================================

    agent.end_episode()

    return (
        steps,
        total_reward,
    )


# =========================================================
# TRAIN MONTE CARLO
# =========================================================
def train_monte_carlo(
    env: LawnMowerEnv,
    agent: MonteCarloControl,
    num_episodes: int = 2000,
    max_steps: int = 500,
):

    episode_lengths = []

    episode_rewards = []

    for episode in range(num_episodes):

        (
            episode_length,
            episode_reward,
        ) = run_monte_carlo_episode(
            env,
            agent,
            max_steps=max_steps,
        )

        episode_lengths.append(
            episode_length
        )

        episode_rewards.append(
            episode_reward
        )

        # =============================================
        # EPSILON DECAY
        # =============================================

        agent.epsilon = max(
            0.05,
            agent.epsilon * 0.995,
        )

        # =============================================
        # LOGGING
        # =============================================

        if (episode + 1) % 100 == 0:

            avg_reward = sum(
                episode_rewards[-100:]
            ) / 100

            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return (
        episode_lengths,
        episode_rewards,
    )


# =========================================================
# RUN EXPERIMENT
# =========================================================
def run_lawn_mower_monte_carlo():

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "lawn_mower_monte_carlo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # ENVIRONMENT
    # =====================================================

    env = LawnMowerEnv(
        L_MAP
    )

    actions = env.possible_actions(
        env.reset()
    )

    # =====================================================
    # AGENT
    # =====================================================

    agent = MonteCarloControl(
        actions=actions,
        gamma=0.99,
        epsilon=0.6,
    )

    agent.reset()

    # =====================================================
    # TRAIN
    # =====================================================

    (
        episode_lengths,
        episode_rewards,
    ) = train_monte_carlo(
        env,
        agent,
        num_episodes=10000,
        max_steps=200,
    )

    # =====================================================
    # TRAINING REWARD CURVE
    # =====================================================
    plot_training_rewards(episode_rewards, output_dir, algorithm_name)

    # =====================================================
    # EPISODE LENGTH CURVE
    # =====================================================
    plot_episode_lengths(episode_lengths, output_dir, algorithm_name)
    
    # =====================================================
    # EXECUTE GREEDY POLICY
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

        if position not in visit_counts:
            visit_counts[position] = 0

        visit_counts[position] += 1

        action = agent.greedy_action(
            state
        )

        next_state, reward, done = (
            env.step(
                state,
                action,
            )
        )

        state = next_state

    # =====================================================
    # FINAL POSITION
    # =====================================================

    position, visited = state

    trajectory.append(
        position
    )

    if position not in visit_counts:
        visit_counts[position] = 0

    visit_counts[position] += 1

    # =====================================================
    # METRICS
    # =====================================================

    coverage_ratio = (
        len(visited)
        /
        env.n_valid_cells
    )

    total_steps = len(
        trajectory
    )

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
    print("MONTE CARLO COVERAGE RESULTS")
    print("======================================")

    print(
        f"Coverage Ratio : "
        f"{coverage_ratio:.2%}"
    )

    print(
        f"Total Steps    : "
        f"{total_steps}"
    )

    print(
        f"Repeated Visits: "
        f"{repeated_visits}"
    )

    print(
        f"Efficiency     : "
        f"{efficiency:.2%}"
    )

    print("======================================")

    # =====================================================
    # FINAL HEATMAP
    # =====================================================
    plot_coverage_heatmap(env, visit_counts, trajectory, position, output_dir, algorithm_name)

    input("Press Enter to close...")
