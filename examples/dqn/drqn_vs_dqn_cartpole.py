from atomic_rl.initialization import layer_init_
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple, List, Optional, Callable, Dict
import numpy as np
import random
import wandb
from tensordict import TensorDict
from functools import partial

from atomic_rl.buffers.replay import (
    init_buffer,
    circular_write_strategy_,
    uniform_sample,
    make_padded_chunk_accumulator,
)
from atomic_rl.losses import mse_loss, with_sequence_mask
from atomic_rl.td import compute_q_td_target
from atomic_rl.action_selection import (
    argmax_selector,
    with_epsilon_greedy,
)
from atomic_rl.schedules import get_linear_schedule
from atomic_rl.optimizer import apply_gradients_
from atomic_rl.bptt.unroll_rnn import unroll_rnn
from atomic_rl.update_target_net import hard_update_target_network_
from envs.wrappers import FlickeringObservation

# TODO: make this be more like PPO + LSTM. First fix the TODOs in PPO + LSTM relating to LSTM stuff.
# TODO: make this use the sequence storage sort of like R2D2.
# TODO: basically this doesnt work at all yet.
