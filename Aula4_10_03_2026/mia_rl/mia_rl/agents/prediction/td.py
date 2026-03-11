
from __future__ import annotations

from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class TD0Prediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, alpha: float = 0.05, gamma: float = 1.0):
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float) # Initialize the value table as a defaultdict of floats, which defaults to 0.0 for unseen states.

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        """Update the value table using TD(0).

        TODO:
        For each transition, compute the TD target:
            reward + gamma * V(next_state)
        Use 0 as the bootstrap value on terminal transitions,
        then apply the incremental TD(0) update with self.alpha.
        """
        for transition in episode.transitions:
            state = transition.state
            reward = transition.reward

            if transition.done or transition.next_state is None:
                bootstrap_value = 0.0
            else:
                bootstrap_value = self.V[transition.next_state]

            td_target = reward + self.gamma * bootstrap_value
            self.V[state] += self.alpha * (td_target - self.V[state])
        #raise NotImplementedError("TODO: implement TD(0) prediction.")

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state]) 
