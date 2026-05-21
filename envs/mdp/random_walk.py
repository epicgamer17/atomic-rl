# TODO: clean up Random Walk MDP, make it cleaner and more general.
import torch


# TODO: should this be a gym env?
class RandomWalkEnv:
    def __init__(self, num_states=5, start_state=2):
        self.num_states = num_states
        self.start_state = start_state
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self._get_features(self.state)

    def step(self):
        # Random transition
        self.state += 1 if torch.rand(1).item() > 0.5 else -1

        terminated = False
        reward = 0.0

        if self.state == self.num_states:  # Right terminal
            reward = 1.0
            terminated = True
        elif self.state == -1:  # Left terminal
            reward = 0.0
            terminated = True

        next_features = (
            self._get_features(self.state)
            if not terminated
            else torch.zeros(self.num_states)
        )

        return next_features, reward, terminated

    def _get_features(self, state):
        phi = torch.zeros(self.num_states)
        if 0 <= state < self.num_states:
            phi[state] = 1.0
        return phi
