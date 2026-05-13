import torch
from typing import Iterator, Tuple


def make_random_walk_episode(
    num_non_terminal_states: int = 5, start_state: int = 2
) -> Iterator[Tuple[torch.Tensor, float, torch.Tensor, bool]]:
    """
    Generates transitions for a single episode of the Random Walk (Sutton 1988).

    Args:
        num_non_terminal_states: Number of non-terminal states (default 5 for A,B,C,D,E).
        start_state: The starting index (default 2 for C).

    Yields:
        Tuple containing:
        - phi_t: One-hot encoded state tensor.
        - reward: Float reward for the transition.
        - phi_next: One-hot encoded next state tensor (zeros if terminated).
        - terminated: Boolean flag indicating episode end.
    """
    state = start_state

    while True:
        phi_t = torch.zeros(num_non_terminal_states)
        phi_t[state] = 1.0

        # Transition: Left (-1) or Right (+1) with equal probability
        next_state = state + (1 if torch.rand(1).item() > 0.5 else -1)

        terminated = False
        reward = 0.0
        phi_next = torch.zeros(num_non_terminal_states)

        if next_state == num_non_terminal_states:  # Right terminal (Win)
            reward = 1.0
            terminated = True
        elif next_state == -1:  # Left terminal (Lose)
            reward = 0.0
            terminated = True
        else:
            phi_next[next_state] = 1.0

        yield phi_t, reward, phi_next, terminated

        if terminated:
            break
