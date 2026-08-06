"""
Baird's Counterexample MDP (Baird 1995, Sutton & Barto 2nd Ed. Ch. 11.2).

Illustrates the 'Deadly Triad' of Reinforcement Learning:
1. Off-Policy Sampling (behavior policy mu vs target policy pi)
2. Function Approximation (linear features phi)
3. Bootstrapping (Temporal Difference learning)

Semi-gradient TD(0) mathematically diverges on this MDP,
while Gradient TD algorithms (GTD0, TDC) converge to V*(s) = 0.
"""

import torch


class BairdsCounterexampleEnv:
    """
    Baird's 7-state counterexample environment.
    States 0..5 (upper states) have feature vector [1, 0, ..., 2 at index i+1, ... 0].
    State 6 (lower state) has feature vector [2, 0, 0, 0, 0, 0, 0, 1].
    """

    def __init__(self):
        self.num_states = 7
        self.num_features = 8
        self.state = 0

        # Feature Matrix Phi (7 x 8)
        self.phi_matrix = torch.tensor([
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])

    def reset(self):
        self.state = torch.randint(0, 7, (1,)).item()
        return self._get_features(self.state)

    def step(self):
        # Action under behavior policy mu:
        # Solid action (prob 1/7): transitions to lower state 6
        # Dashed action (prob 6/7): transitions to upper state (0..5)
        is_solid = torch.rand(1).item() < (1.0 / 7.0)

        if is_solid:
            self.state = 6
            rho = 7.0 # pi(solid)/mu(solid) = 1.0 / (1/7) = 7.0
        else:
            self.state = torch.randint(0, 6, (1,)).item()
            rho = 0.0 # pi(dashed)/mu(dashed) = 0.0 / (6/7) = 0.0

        reward = 0.0
        terminated = False

        next_features = self._get_features(self.state)
        return next_features, reward, terminated, rho

    def _get_features(self, state: int):
        return self.phi_matrix[state].clone()
