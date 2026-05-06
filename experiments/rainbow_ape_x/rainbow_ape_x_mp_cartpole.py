"""
APE-X with Rainbow (Dueling + Noisy + Distributional) implemented with torch.multiprocessing.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import random
import wandb
from functools import partial
from typing import Tuple, Dict, Any, Callable, List

from functional.buffer import (
    init_per_buffer,
    sample_per,
    update_priorities,
    with_per_tracking,
    circular_write_strategy,
    make_n_step_accumulator,
    PERBufferState,
)
from functional.losses import bellman_error, cross_entropy_loss
from functional.targets import categorical_td_target
from functional.action_selection import (
    double_selector,
    categorical_extractor,
    get_ape_x_epsilon,
)
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network
from networks.noisy_linear import NoisyLinear

# --- Constants ---
ENV_NAME = "CartPole-v1"
SEED = 42

BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 1e-3
BUFFER_CAPACITY = 50_000
MIN_BUFFER_SIZE = 5_000
TARGET_NET_UPDATE_FREQ = 100
MAX_LEARNER_STEPS = 30_000

# Ape-X Epsilon parameters (used for Noisy Sigma)
BASE_EPSILON = 0.5
EPSILON_ALPHA = 1.0

NUM_ACTORS = 4
ACTOR_BATCH_SIZE = 500

ALPHA = 0.6
BETA = 0.4
N_STEP = 3
MAX_GRAD_NORM = 10.0

# Distributional DQN (C51) Constants
V_MIN = 0.0
V_MAX = 500.0
ATOM_SIZE = 51

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class RainbowDQN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        atom_size: int,
        support: torch.Tensor,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.atom_size = atom_size
        self.register_buffer("support", support)

        self.feature_layer = NoisyLinear(input_shape[0], 512, sigma_init=sigma_init)

        self.advantage_head = nn.Sequential(
            NoisyLinear(512, 512, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions * atom_size, sigma_init=sigma_init),
        )

        self.value_head = nn.Sequential(
            NoisyLinear(512, 512, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(512, atom_size, sigma_init=sigma_init),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.feature_layer(x))
        advantage = self.advantage_head(x).view(-1, self.num_actions, self.atom_size)
        value = self.value_head(x).view(-1, 1, self.atom_size)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return logits

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class SharedPERBuffer:
    def __init__(self, capacity: int, shapes: Dict[str, Any]):
        self.buffer_state = init_per_buffer(capacity, shapes)
        self.buffer_state.data.share_memory_()
        self.buffer_state.sum_tree.share_memory_()
        self.buffer_state.min_tree.share_memory_()

        self.lock = mp.Lock()

        self._pointer = mp.Value("i", 0)
        self._size = mp.Value("i", 0)
        self._max_priority = mp.Value("d", 1.0)

        # This is a local function from with_per_tracking, which is not picklable.
        # We'll initialize it in __init__ and __setstate__.
        self.per_add = with_per_tracking(circular_write_strategy)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the non-picklable local function
        if "per_add" in state:
            del state["per_add"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize the local function after unpickling
        self.per_add = with_per_tracking(circular_write_strategy)

    def add_transitions(self, transitions_dict: Dict[str, torch.Tensor]):
        with self.lock:
            self.buffer_state.pointer = self._pointer.value
            self.buffer_state.size = self._size.value
            self.buffer_state.max_priority = self._max_priority.value

            self.buffer_state = self.per_add(self.buffer_state, transitions_dict)

            self._pointer.value = self.buffer_state.pointer
            self._size.value = self.buffer_state.size
            self._max_priority.value = self.buffer_state.max_priority

    def sample_batch(
        self, batch_size: int, beta: float, min_size: int = 0
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        with self.lock:
            if self._size.value < min_size:
                return None, None, None
            self.buffer_state.size = self._size.value
            beta_tensor = torch.tensor(beta, dtype=torch.float32)
            return sample_per(self.buffer_state, batch_size, beta_tensor)

    def update_priorities(
        self, tree_indices: torch.Tensor, td_errors: torch.Tensor, alpha: float
    ):
        with self.lock:
            self.buffer_state = update_priorities(
                self.buffer_state, tree_indices, td_errors, alpha
            )
            self._max_priority.value = self.buffer_state.max_priority


def actor_worker(
    actor_id: int,
    num_actors: int,
    model_creator: Callable[[float], nn.Module],
    shared_model: nn.Module,
    buffer: SharedPERBuffer,
    selector_fn: Callable,
    target_fn: Callable,
    loss_fn: Callable,
    support: torch.Tensor,
):
    env = gym.make(ENV_NAME)
    sigma_init = get_ape_x_epsilon(
        actor_id, num_actors, base_eps=BASE_EPSILON, alpha=EPSILON_ALPHA
    )

    local_model = model_creator(sigma_init)
    local_target_model = model_creator(sigma_init)

    local_model.load_state_dict(shared_model.state_dict())
    local_target_model.load_state_dict(shared_model.state_dict())

    n_step_proc, n_step_reset = make_n_step_accumulator(N_STEP, GAMMA)
    obs, _ = env.reset(seed=SEED + actor_id)
    episode_return = 0.0
    step_count = 0
    local_batch = []

    wandb.init(
        project="rainbow-ape-x-mp-cartpole",
        group="ape-x-distributed",
        job_type=f"actor_{actor_id}",
        config={"actor_id": actor_id, "sigma_init": sigma_init},
    )

    while True:
        if step_count % 100 == 0:
            local_model.load_state_dict(shared_model.state_dict())
            local_target_model.load_state_dict(shared_model.state_dict())

        local_model.reset_noise()

        with torch.inference_mode():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            _, actions = selector_fn(local_model, None, obs_tensor)
            action = actions.item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward

        n_step_transitions = n_step_proc(
            obs, action, reward, next_obs, terminated, truncated
        )
        for n_step_td in n_step_transitions:
            local_batch.append(n_step_td)

        obs = next_obs
        step_count += 1

        if terminated or truncated:
            wandb.log(
                {
                    f"actor_{actor_id}/episode_return": episode_return,
                    "global/episode_return": episode_return,
                },
                step=step_count,
            )
            obs, _ = env.reset()
            episode_return = 0.0
            n_step_reset()

        if len(local_batch) >= ACTOR_BATCH_SIZE:
            collated = {
                k: torch.cat([t[k] for t in local_batch]) for k in local_batch[0].keys()
            }
            with torch.no_grad():
                _, info_dict = bellman_error(
                    local_model,
                    collated,
                    local_model,
                    partial(target_fn, gamma=collated["gamma"]),
                    eval_model=local_target_model,
                    loss_fn=loss_fn,
                )
            collated["priority"] = info_dict["priorities"].unsqueeze(-1)
            buffer.add_transitions(collated)
            local_batch = []


def learner_worker(
    model_creator: Callable[[float], nn.Module],
    shared_model: nn.Module,
    buffer: SharedPERBuffer,
    selector_fn: Callable,
    target_fn: Callable,
    loss_fn: Callable,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_model = model_creator(0.5).to(device)
    target_model.load_state_dict(shared_model.state_dict())

    local_model = model_creator(0.5).to(device)
    local_model.load_state_dict(shared_model.state_dict())

    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

    wandb.init(
        project="rainbow-ape-x-mp-cartpole",
        group="ape-x-distributed",
        job_type="learner",
        config={
            "batch_size": BATCH_SIZE,
            "num_actors": NUM_ACTORS,
            "learning_rate": LEARNING_RATE,
        },
    )

    step_count = 0
    while step_count < MAX_LEARNER_STEPS:
        result = buffer.sample_batch(BATCH_SIZE, BETA, MIN_BUFFER_SIZE)
        if result[0] is None:
            time.sleep(0.1)
            continue

        batch, indices, is_weights = result
        batch = {k: v.to(device) for k, v in batch.items()}
        indices = indices.to(device)
        is_weights = is_weights.to(device)

        local_model.reset_noise()
        target_model.reset_noise()

        loss, info = bellman_error(
            local_model,
            batch,
            local_model,
            partial(target_fn, gamma=batch["gamma"]),
            eval_model=target_model,
            loss_fn=loss_fn,
        )

        weighted_loss = (loss * is_weights).mean()
        optimizer = apply_gradients(
            optimizer,
            weighted_loss,
            model=local_model,
            clip_grad_norm=MAX_GRAD_NORM,
        )

        shared_model.load_state_dict(local_model.state_dict())

        td_errors = info["priorities"].detach().cpu()
        buffer.update_priorities(indices.cpu(), td_errors, ALPHA)

        step_count += 1
        if step_count % TARGET_NET_UPDATE_FREQ == 0:
            hard_update_target_network(local_model, target_model)

        if step_count % 100 == 0:
            print(f"Step {step_count}: Loss={weighted_loss.item():.4f}")
            wandb.log(
                {
                    "learner/loss": weighted_loss.item(),
                    "learner/mean_q": info["q_values/mean"],
                },
                step=step_count,
            )


def rainbow_model_creator_fn(sigma, obs_shape, num_actions, atom_size, support):
    return RainbowDQN(obs_shape, num_actions, atom_size, support, sigma_init=sigma)


def main():
    # Set start method for multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    temp_env = gym.make(ENV_NAME)
    obs_shape = temp_env.observation_space.shape
    num_actions = temp_env.action_space.n
    temp_env.close()

    support = torch.linspace(V_MIN, V_MAX, ATOM_SIZE)

    my_model_creator = partial(
        rainbow_model_creator_fn,
        obs_shape=obs_shape,
        num_actions=num_actions,
        atom_size=ATOM_SIZE,
        support=support,
    )

    my_selector_fn = partial(
        double_selector,
        extractor_fn=partial(categorical_extractor, support=support),
    )
    my_target_fn = partial(
        categorical_td_target,
        support=support,
        v_min=V_MIN,
        v_max=V_MAX,
        atom_size=ATOM_SIZE,
    )
    my_loss_fn = cross_entropy_loss

    shared_model = my_model_creator(0.5)
    shared_model.share_memory()

    buffer_shapes = {
        "obs": obs_shape,
        "action": (1,),
        "reward": (1,),
        "terminated": (1,),
        "truncated": (1,),
        "next_obs": obs_shape,
        "gamma": (1,),
        "priority": (1,),
    }
    buffer = SharedPERBuffer(BUFFER_CAPACITY, buffer_shapes)

    processes = []

    p = mp.Process(
        target=learner_worker,
        args=(
            my_model_creator,
            shared_model,
            buffer,
            my_selector_fn,
            my_target_fn,
            my_loss_fn,
        ),
    )
    p.start()
    processes.append(p)

    for i in range(NUM_ACTORS):
        p = mp.Process(
            target=actor_worker,
            args=(
                i,
                NUM_ACTORS,
                my_model_creator,
                shared_model,
                buffer,
                my_selector_fn,
                my_target_fn,
                my_loss_fn,
                support,
            ),
        )
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Interrupting...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
