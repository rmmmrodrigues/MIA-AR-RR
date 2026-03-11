from __future__ import annotations

from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class FirstVisitMonteCarloPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def reset(self) -> None:
        self.V = defaultdict(float) #estimated value of the state
        self.N = defaultdict(int) #how many times have I already seen this state as a first visit?

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        """Update the value table using first-visit Monte Carlo prediction.

        TODO:
        1. Traverse the episode backwards to compute the return G for each time step.
        2. Identify the first visit of each state within the episode.
        3. Update the sample-average estimate using self.N[state].
            # Incremental mean update:
            # V_new = V_old + (G - V_old) / N
        """
        returns = [0.0] * len(episode.transitions)
        G = 0.0

        # 1) Backward pass to compute returns G_t
        for t in range(len(episode.transitions) - 1, -1, -1):
            transition = episode.transitions[t]
            G = transition.reward + self.gamma * G
            returns[t] = G

        # 2) First-visit updates only
        seen_states = dict[BlackjackState, 1]()
        for t, transition in enumerate(episode.transitions):
            seen_states.setdefault(transition.state, t)  # Record the first visit index for each state
            #state = transition.state
            #if state in seen_states:
            #    continue

            #seen_states.add(state)
            #self.N[state] += 1
            #self.V[state] += (returns[t] - self.V[state]) / self.N[state]
        
        for t, transition in enumerate(episode.transitions):
            state = transition.state
            if state not in seen_states:
                seen_states[state] = True
                self.N[state] += 1
                self.V[state] += (returns[t] - self.V[state]) / self.N[state]
        #raise NotImplementedError("TODO: implement first-visit Monte Carlo prediction.")
    
        # # 1) Backward pass to compute returns G_t
        # for t in range(len(episode.transitions) - 1, -1, -1):
        #     transition = episode.transitions[t]
        #     G = transition.reward + self.gamma * G
        #     returns[t] = G

        # # 2) First-visit updates only
        # seen_states = set()
        # for t, transition in enumerate(episode.transitions):
        #     state = transition.state
        #     if state in seen_states:
        #         continue

        #     seen_states.add(state)
        #     self.N[state] += 1
        #     self.V[state] += (returns[t] - self.V[state]) / self.N[state]

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])  #the agent’s current estimate for that state
