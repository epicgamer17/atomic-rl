from pettingzoo.classic import tictactoe_v3
from atomic_rl.envs.wrappers import (
    ActionMaskInInfoWrapper,
    ChannelLastToFirstWrapper,
    FrameStackWrapper,
    TwoPlayerPlayerPlaneWrapper,
)


def tictactoe_factory(render_mode=None):
    """
    Creates a TicTacToe environment with standard wrappers for RL training.

    Wrappers applied:
    - ActionMaskInInfoWrapper: Moves action mask to info['legal_moves']
    - FrameStackWrapper (k=4): Stacks last 4 frames
    - TwoPlayerPlayerPlaneWrapper: Adds indicator plane for whose turn it is
    - ChannelLastToFirstWrapper: Swaps HWC to CHW
    """
    env = tictactoe_v3.env(render_mode=render_mode)
    env = ActionMaskInInfoWrapper(env)
    # Apply FrameStack and PlayerPlane on channel-last initially, then swap to first
    env = FrameStackWrapper(env, k=4, channel_first=False)
    env = TwoPlayerPlayerPlaneWrapper(env, channel_first=False)
    env = ChannelLastToFirstWrapper(env)
    return env
