#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


# ============================================================
# Environment: multi-armed bandit
# ============================================================

class KArmedBandit:
    def __init__(self, k=10, stationary=True, walk_std=0.01):
        self.k = k
        self.stationary = stationary
        self.walk_std = walk_std
        self.reset()

    def reset(self):
        self.q_true = np.random.randn(self.k)  # true action values
        self.optimal_action = np.argmax(self.q_true)

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]

        # non-stationary random walk
        if not self.stationary:
            self.q_true += np.random.normal(0, self.walk_std, self.k)
            self.optimal_action = np.argmax(self.q_true)

        return reward
