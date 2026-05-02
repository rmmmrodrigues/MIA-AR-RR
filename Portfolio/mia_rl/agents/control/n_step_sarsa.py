from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class NStepSarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        n_steps: int = 4,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")

        self.actions = actions
        self.n_steps = n_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}
        self._transitions: list[Transition[StateT, ActionT]] = []

    def select_action(self, state: StateT) -> ActionT:
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            best_value = max(self.action_value_of(state, action) for action in self.actions)
            best_actions = [action for action in self.actions if self.action_value_of(state, action) == best_value]
            action = self.rng.choice(best_actions)

        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        self._transitions.append(transition)

        if len(self._transitions) >= self.n_steps:
            self._update_oldest(use_bootstrap=True)

        if transition.done:
            self.end_episode()

    def end_episode(self) -> None:
        while self._transitions:
            self._update_oldest(use_bootstrap=False)

    def _update_oldest(self, use_bootstrap: bool) -> None:
        horizon = min(self.n_steps, len(self._transitions))
        target = 0.0

        for idx in range(horizon):
            target += (self.gamma**idx) * self._transitions[idx].reward

        last_transition = self._transitions[horizon - 1]
        if use_bootstrap and not last_transition.done and last_transition.next_state is not None:
            next_action = self._selected_actions[last_transition.next_state]
            target += (self.gamma**horizon) * self.action_value_of(last_transition.next_state, next_action)

        first_transition = self._transitions.pop(0)
        current_value = self.action_value_of(first_transition.state, first_transition.action)
        self.Q[(first_transition.state, first_transition.action)] = current_value + self.alpha * (
            target - current_value
        )

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))
