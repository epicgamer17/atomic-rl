"""
APE-X DQN implemented with torch.multiprocessing.
"""

# TODO: attempt a cleanup if possible

from functional.initialization import layer_init
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
from tensordict import TensorDict
from functools import partial
from typing import Tuple, Dict, Any, Callable, List

from functional.replay_buffer import (
    init_per_buffer,
    sample_per,
    update_priorities,
    with_per_tracking,
    circular_write_strategy,
    make_n_step_accumulator,
    PERBufferState,
)
from functional.losses import huber_loss
from functional.td import compute_q_td_target
from functional.action_selection import (
    argmax_selector,
    gather_q_values,
)
from functional.schedules import get_ape_x_epsilon
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network

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

# 0.2 and 1.0 because carpole is sensitive to bad and random moves
BASE_EPSILON = 0.2
EPSILON_ALPHA = 1.0

NUM_ACTORS = 4
ACTOR_BATCH_SIZE = 500

ALPHA = 0.6
BETA = 0.4
N_STEP = 3
MAX_GRAD_NORM = 10.0

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = layer_init(nn.Linear(input_shape[0], 512))
        self.value_head = layer_init(nn.Linear(512, 1), std=1.0)
        self.advantage_head = layer_init(nn.Linear(512, num_actions), std=1.0)

    def forward(self, x):
        x = F.relu(self.l1(x))
        v = self.value_head(x)
        a = self.advantage_head(x)
        return v + a - a.mean(dim=1, keepdim=True)


class SharedPERBuffer:
    def __init__(self, capacity: int, shapes: Dict[str, Any]):
        self.buffer_state = init_per_buffer(capacity, shapes)
        # Move all tensors to shared memory
        self.buffer_state.data.share_memory_()
        self.buffer_state.sum_tree.share_memory_()
        self.buffer_state.min_tree.share_memory_()

        self.lock = mp.Lock()

        # We need to share the pointer and size too.
        # PERBufferState is a dataclass, so we might need to wrap its scalars in mp.Value if we want them shared.
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

    def add_transitions(self, transitions: TensorDict):
        with self.lock:
            # Sync state from shared scalars
            self.buffer_state.pointer = self._pointer.value
            self.buffer_state.size = self._size.value
            self.buffer_state.max_priority = self._max_priority.value

            self.buffer_state = self.per_add(self.buffer_state, transitions)

            # Sync back to shared scalars
            self._pointer.value = self.buffer_state.pointer
            self._size.value = self.buffer_state.size
            self._max_priority.value = self.buffer_state.max_priority

    def sample_batch(
        self, batch_size: int, beta: float, min_size: int = 0
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        with self.lock:
            if self._size.value < min_size:
                return None, None, None

            # Sync state
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
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedPERBuffer,
    selector_fn: Callable,
    target_fn: Callable,
    loss_fn: Callable,
):
    # Setup environment
    env = gym.make(ENV_NAME)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    epsilon = get_ape_x_epsilon(
        actor_id, num_actors, base_eps=BASE_EPSILON, alpha=EPSILON_ALPHA
    )

    # Local models
    local_model = model_creator()
    local_target_model = model_creator()
    num_actions = env.action_space.n

    # Initial sync
    local_model.load_state_dict(shared_model.state_dict())
    local_target_model.load_state_dict(shared_model.state_dict())

    n_step_proc, n_step_reset = make_n_step_accumulator(N_STEP, GAMMA)
    obs, _ = env.reset(seed=SEED + actor_id)
    step_count = 0
    local_batch = []

    wandb.init(
        project="ape-x-mp-cartpole",
        group="ape-x-distributed",
        job_type=f"actor_{actor_id}",
        config={"actor_id": actor_id, "epsilon": epsilon},
    )

    while True:
        # Periodic weight sync
        if step_count % 100 == 0:
            local_model.load_state_dict(shared_model.state_dict())
            local_target_model.load_state_dict(shared_model.state_dict())

        # Act
        with torch.inference_mode():
            obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32)
            predictions = local_model(obs_tensor)
            greedy_actions, _ = argmax_selector(predictions)
            # TODO: Remove this and use epsilon-greedy selector function.
            if random.random() < epsilon:
                action_np = random.randint(0, num_actions - 1)
                action = torch.tensor([action_np], dtype=torch.long)
            else:
                action_np = greedy_actions.item()
                action = greedy_actions.squeeze(0).detach().to(torch.long)

        # Step
        # Extract the scalar for a non-vectorized Gymnasium environment
        action_int = int(action_np.item())
        next_obs, reward, terminated, truncated, info = env.step(action_int)

        # TODO: URGENT. Creating a new tensor every step in the hotloop. should do in a more efficient way, maybe some thing like the rollout buffer in PPO (possible reuse?) ie store in a rollout buffer before sending to main replay buffer. Idea being its pre allocated basically. Must consider the N-Step case.
        n_step_transitions = n_step_proc(
            obs, action, reward, next_obs, terminated, truncated
        )
        for n_step_td in n_step_transitions:
            local_batch.append(n_step_td)

        obs = next_obs
        step_count += 1

        if terminated or truncated:
            if "episode" in info:
                wandb.log(
                    {
                        f"actor_{actor_id}/episode_return": info["episode"]["r"][0],
                        f"actor_{actor_id}/episode_length": info["episode"]["l"][0],
                        "global/episode_return": info["episode"]["r"][0],
                    },
                    step=step_count,
                )
            obs, _ = env.reset()
            n_step_reset()

        # Push to buffer
        if len(local_batch) >= ACTOR_BATCH_SIZE:
            collated = TensorDict(
                {
                    k: torch.cat([t[k] for t in local_batch])
                    for k in local_batch[0].keys()
                },
                batch_size=[len(local_batch)],
            )
            with torch.no_grad():
                q_values = local_model(collated["obs"])
                next_q_values_online = local_model(collated["next_obs"])
                next_q_values_target = local_target_model(collated["next_obs"])

                next_actions, _ = argmax_selector(next_q_values_online)

                td_target = compute_q_td_target(
                    next_q_values_target,
                    next_actions.squeeze(-1),
                    collated["reward"],
                    collated["terminated"],
                    collated["gamma"],
                )
                pred_sa = gather_q_values(q_values, collated["action"])
                _, info_dict = loss_fn(pred_sa, td_target)
            collated["priority"] = info_dict["priorities"]
            buffer.add_transitions(collated)
            local_batch = []


def learner_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedPERBuffer,
    selector_fn: Callable,
    target_fn: Callable,
    loss_fn: Callable,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Learner's local target model
    target_model = model_creator().to(device)
    target_model.load_state_dict(shared_model.state_dict())

    # Learner uses a local copy of the model for training, then pushes to shared_model
    # Actually, shared_model is in shared memory, so we can optimize it directly if we move it to device.
    # But for simplicity and to follow the Ape-X pattern, let's keep a local optimizer and push.
    # If shared_model is on CPU, we definitely want a local GPU model.

    local_model = model_creator().to(device)
    local_model.load_state_dict(shared_model.state_dict())

    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

    wandb.init(
        project="ape-x-mp-cartpole",
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
        # Sample
        result = buffer.sample_batch(BATCH_SIZE, BETA, MIN_BUFFER_SIZE)
        if result[0] is None:
            time.sleep(0.1)
            continue

        batch, indices, is_weights = result
        batch = batch.to(device)
        indices = indices.to(device)
        is_weights = is_weights.to(device)

        # 1. Forward Passes (Online and Target)
        q_values = local_model(batch["obs"])
        with torch.no_grad():
            next_q_values_online = local_model(batch["next_obs"])
            next_q_values_target = target_model(batch["next_obs"])

            # 2. Next Action Selection (Double DQN: Online model selects)
            next_actions, _ = argmax_selector(next_q_values_online)

            # 3. Target Calculation
            td_target = compute_q_td_target(
                next_q_values_target,
                next_actions.squeeze(-1),
                batch["reward"],
                batch["terminated"],
                batch["gamma"],
            )

        # 4. Prediction Extraction
        pred_sa = gather_q_values(q_values, batch["action"])

        # 5. Loss Calculation
        raw_loss, info = loss_fn(pred_sa, td_target)
        weighted_loss = (raw_loss * is_weights).mean()

        # Augment info for logging
        info.update(
            {
                "q_values/mean": pred_sa.mean().detach(),
                "td_targets/mean": td_target.mean().detach(),
            }
        )
        optimizer = apply_gradients(
            optimizer,
            weighted_loss,
            model=local_model,
            clip_grad_norm=MAX_GRAD_NORM,
        )

        # Update shared model
        shared_model.load_state_dict(local_model.state_dict())

        # Update priorities
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


def model_creator_fn(obs_shape, num_actions):
    return DuelingDQN(obs_shape, num_actions)


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

    my_model_creator = partial(
        model_creator_fn, obs_shape=obs_shape, num_actions=num_actions
    )
    my_selector_fn = argmax_selector
    my_target_fn = compute_q_td_target
    my_loss_fn = huber_loss

    # Shared model for weight syncing
    shared_model = my_model_creator()
    shared_model.share_memory()

    # Shared buffer
    buffer_shapes = {
        "obs": obs_shape,
        "action": (1,),
        "reward": (),
        "terminated": (),
        "truncated": (),
        "next_obs": obs_shape,
        "gamma": (),
        "priority": (),
    }
    buffer = SharedPERBuffer(BUFFER_CAPACITY, buffer_shapes)

    processes = []

    # Start Learner
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

    # Start Actors
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
