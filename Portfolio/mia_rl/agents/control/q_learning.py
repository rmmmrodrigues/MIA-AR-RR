from __future__ import annotations

from collections import defaultdict
import random

# =========================================================
# Q-LEARNING AGENT
# =========================================================

class QLearningAgent:

    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):

        self.env = env

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q[state][action]
        self.Q = defaultdict(lambda: defaultdict(float))

    # =====================================================
    # ACTION SELECTION
    # =====================================================

    def choose_action(self, state):

        actions = self.env.possible_actions(state)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(actions)

        q_values = self.Q[state]

        best_action = max(
            actions,
            key=lambda a: q_values[a]
        )

        return best_action

    # =====================================================
    # GREEDY ACTION
    # =====================================================

    def greedy_action(self, state):

        actions = self.env.possible_actions(state)

        q_values = self.Q[state]

        best_action = max(
            actions,
            key=lambda a: q_values[a]
        )

        return best_action

    # =====================================================
    # TRAIN
    # =====================================================

    def train(
        self,
        episodes=1000,
        max_steps=1000,
    ):

        rewards_history = []

        for episode in range(episodes):

            state = self.env.reset()

            total_reward = 0.0

            for step in range(max_steps):

                action = self.choose_action(state)

                next_state, reward, done = self.env.step(
                    state,
                    action
                )

                # =========================================
                # Q-LEARNING UPDATE
                # =========================================

                next_actions = self.env.possible_actions(next_state)

                if len(next_actions) > 0:

                    best_next_q = max(
                        self.Q[next_state][a]
                        for a in next_actions
                    )

                else:

                    best_next_q = 0.0

                current_q = self.Q[state][action]

                new_q = (
                    current_q
                    +
                    self.alpha * (
                        reward
                        +
                        self.gamma * best_next_q
                        -
                        current_q
                    )
                )

                self.Q[state][action] = new_q

                state = next_state

                total_reward += reward

                if done:
                    break

            # =============================================
            # EPSILON DECAY
            # =============================================

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

            rewards_history.append(total_reward)

            # =============================================
            # LOG
            # =============================================

            if (episode + 1) % 100 == 0:

                print(
                    f"Episode {episode+1} | "
                    f"Reward = {total_reward:.2f} | "
                    f"Epsilon = {self.epsilon:.3f}"
                )

        return rewards_history