from __future__ import annotations

import numpy as np

# ============================================================
# Base agent
# ============================================================

class BanditAgent:
    def __init__(self, k=10):
        self.k = k
        self.reset()

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.t = 0

    def select_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        raise NotImplementedError


# ============================================================
# ε-greedy agent
# ============================================================

class EpsilonGreedy(BanditAgent):
    def __init__(self, k=10, epsilon=0.1, alpha=None, optimistic=0.0):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.optimistic = optimistic
        self.reset()

    def reset(self):
        super().reset()
        self.Q[:] = self.optimistic

    def select_action(self):
        if np.random.rand() < self.epsilon:
            # exploração: ação aleatória
            return np.random.randint(self.k)
        else:
            # exploração: melhor ação conhecida
            return np.argmax(self.Q)

    def update(self, action, reward):
        self.t += 1
        self.N[action] += 1

        if self.alpha is None:
            # média incremental
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
        else:
            # passo constante
            self.Q[action] += self.alpha * (reward - self.Q[action])


# ============================================================
# UCB agent
# ============================================================

class UCB(BanditAgent):
    def __init__(self, k=10, c=2.0):
        super().__init__(k)
        self.c = c

    def select_action(self):
        self.t += 1

        # garantir que todas as ações são testadas
        for a in range(self.k):
            if self.N[a] == 0:
                return a

        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


# ============================================================
# Gradient bandit agent
# ============================================================

class GradientBandit(BanditAgent):
    def __init__(self, k=10, alpha=0.1, baseline=True):
        self.k = k
        self.alpha = alpha
        self.baseline = baseline
        self.reset()

    def reset(self):
        super().reset()
        self.H = np.zeros(self.k)
        self.avg_reward = 0.0

    def _policy(self):
        exp = np.exp(self.H - np.max(self.H))
        return exp / np.sum(exp)

    def select_action(self):
        probs = self._policy()
        return np.random.choice(self.k, p=probs)

    def update(self, action, reward):
        self.t += 1
        probs = self._policy()

        if self.baseline:
            self.avg_reward += (reward - self.avg_reward) / self.t
            baseline = self.avg_reward
        else:
            baseline = 0

        for a in range(self.k):
            if a == action:
                self.H[a] += self.alpha * (reward - baseline) * (1 - probs[a])
            else:
                self.H[a] -= self.alpha * (reward - baseline) * probs[a]
