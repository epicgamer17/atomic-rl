from functional.initialization import layer_init
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

from functional.replay_buffer import (
    init_buffer,
    circular_write_strategy,
    uniform_sample,
    make_padded_chunk_accumulator,
)
from functional.losses import mse_loss, with_sequence_mask
from functional.td import compute_q_td_target
from functional.action_selection import (
    argmax_selector,
    with_epsilon_greedy,
)
from functional.schedules import get_linear_schedule
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network_, unroll_rnn
from envs.wrappers import FlickeringObservation

# TODO: make this be more like PPO + LSTM. First fix the TODOs in PPO + LSTM relating to LSTM stuff.
# TODO: make this use the sequence storage sort of like R2D2.
# TODO: basically this doesnt work at all yet.
