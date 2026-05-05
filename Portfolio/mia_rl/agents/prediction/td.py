from __future__ import annotations

from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class TD0Prediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, alpha: float = 0.05, gamma: float = 1.0):
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        for transition in episode.transitions:
            bootstrap = 0.0 if transition.done or transition.next_state is None else self.V[transition.next_state]
            target = transition.reward + self.gamma * bootstrap
            self.V[transition.state] += self.alpha * (target - self.V[transition.state])

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])



class TDNPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, n: int = 3, alpha: float = 0.05, gamma: float = 1.0):
        self.n = n
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        T = len(episode.transitions)

        for t in range(T):
            G = 0.0

            # soma dos rewards até n passos ou fim do episódio
            for k in range(t, min(t + self.n, T)):
                transition = episode.transitions[k]
                G += (self.gamma ** (k - t)) * transition.reward

                if transition.done:
                    break

            # bootstrap se ainda houver estado futuro
            if t + self.n < T:
                next_transition = episode.transitions[t + self.n]
                if not next_transition.done and next_transition.next_state is not None:
                    G += (self.gamma ** self.n) * self.V[next_transition.state]

            state = episode.transitions[t].state
            self.V[state] += self.alpha * (G - self.V[state])

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])