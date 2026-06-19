"""
APE-X with Rainbow:
Idea: Rainbow was good, APE-X was good, they are compatible so why not combine them.
"""

import os
import time
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
import wandb
from functools import partial
from typing import Tuple, Dict, Any, Callable

from functional.replay_buffer import (
    init_per_buffer,
    sample_per,
    update_priorities,
    with_per_tracking,
    circular_write_strategy,
    make_n_step_accumulator,
)
from functional.losses import cross_entropy_loss
from functional.td import compute_categorical_q_td_target
from functional.action_selection import (
    gather_q_values,
    expected_value,
    argmax_selector,
)
from functional.schedules import get_ape_x_epsilon
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
MIN_BUFFER_SIZE = 500
TARGET_NET_UPDATE_FREQ = 100
MAX_LEARNER_STEPS = 30_000

# Ape-X Epsilon parameters (now used for Noisy Sigma)
BASE_EPSILON = 0.5
EPSILON_ALPHA = 1.0

NUM_ACTORS = 4
ACTOR_BATCH_SIZE = 500

ALPHA = 0.6
BETA = 0.4
PREFETCH_COUNT = 3
N_STEP = 3
MAX_GRAD_NORM = 10.0

# Distributional DQN (C51) Constants
V_MIN = 0.0
V_MAX = 500.0
ATOM_SIZE = 51

# Set seeds for all backends
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class RainbowDQN(nn.Module):
    """
    Rainbow DQN Architecture: Dueling + Noisy + Distributional.
    Matches the architecture in examples/dqn/rainbow_dqn_cartpole.py
    """

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

        # Shared feature extractor (Noisy)
        self.feature_layer = NoisyLinear(input_shape[0], 512, sigma_init=sigma_init)

        # Advantage stream
        self.advantage_head = nn.Sequential(
            NoisyLinear(512, 512, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions * atom_size, sigma_init=sigma_init),
        )

        # Value stream
        self.value_head = nn.Sequential(
            NoisyLinear(512, 512, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(512, atom_size, sigma_init=sigma_init),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returns the logits of the categorical distribution.
        Shape: [Batch, Actions, Atoms]
        """
        x = F.relu(self.feature_layer(x))

        advantage = self.advantage_head(x).view(-1, self.num_actions, self.atom_size)
        value = self.value_head(x).view(-1, 1, self.atom_size)

        # Dueling combination
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return logits

    def reset_noise(self):
        """Resets the noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity: int, shapes: Dict[str, Any]):
        self.buffer_state = init_per_buffer(capacity, shapes)
        self.per_add = with_per_tracking(circular_write_strategy)

    def add_transitions(self, transitions_dict: Dict[str, torch.Tensor]):
        self.buffer_state = self.per_add(self.buffer_state, transitions_dict)

    def sample_batch(
        self, batch_size: int, beta: float, min_size: int = 0
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        if self.buffer_state.size < min_size:
            return None, None, None
        beta_tensor = torch.tensor(beta, dtype=torch.float32)
        return sample_per(self.buffer_state, batch_size, beta_tensor)

    def update_priorities(
        self, tree_indices: torch.Tensor, td_errors: torch.Tensor, alpha: float
    ):
        self.buffer_state = update_priorities(
            self.buffer_state, tree_indices, td_errors, alpha
        )


@ray.remote
class ActorActor:
    def __init__(
        self,
        actor_id: int,
        num_actors: int,
        model_creator: Callable[[float], nn.Module],
        target_fn: Callable,
        loss_fn: Callable,
        learner: ray.actor.ActorHandle,
        buffer: ray.actor.ActorHandle,
        support: torch.Tensor,
    ):
        self.actor_id = actor_id
        self.env = gym.make(ENV_NAME)
        self.device = torch.device("cpu")

        # Ape-X Epsilon used as initial sigma for Noisy Nets
        self.sigma_init = get_ape_x_epsilon(
            actor_id, num_actors, base_eps=BASE_EPSILON, alpha=EPSILON_ALPHA
        )

        self.learner = learner
        self.buffer = buffer
        self.support = support.to(self.device)

        # Model created with unique sigma_init
        self.model = model_creator(self.sigma_init).to(self.device)
        self.target_model = model_creator(self.sigma_init).to(self.device)

        self.model.train()
        self.target_model.train()

        self.num_actions = self.env.action_space.n
        self.target_fn = partial(
            target_fn,
            support=self.support,
            v_min=V_MIN,
            v_max=V_MAX,
            atom_size=ATOM_SIZE,
        )
        self.loss_fn = loss_fn
        self.local_batch = []
        self.obs, _ = self.env.reset(seed=SEED + actor_id)
        self.episode_return = 0.0
        self.step_count = 0
        self.n_step_proc, self.n_step_reset = make_n_step_accumulator(N_STEP, GAMMA)
        self.weights_ref = self.learner.get_weights.remote()

        wandb.init(
            project="rainbow-ape-x-ray-cartpole",
            group="ape-x-distributed",
            job_type=f"actor_{actor_id}",
            config={
                "actor_id": actor_id,
                "sigma_init": self.sigma_init,
                "noisy_nets": True,
            },
            mode="online",
        )
        print(f"Actor {actor_id} initialized with sigma_init={self.sigma_init:.4f}")

    def run(self):
        print(f"Actor {self.actor_id} starting run loop...")
        while True:
            # 1. Non-blocking weight sync
            if self.weights_ref is not None:
                ready, _ = ray.wait([self.weights_ref], timeout=0)
                if ready:
                    weights = ray.get(self.weights_ref)
                    self.model.load_state_dict(weights)
                    self.target_model.load_state_dict(weights)
                    self.weights_ref = None

            # Only request new weights every 100 steps if no request is pending
            if self.weights_ref is None and self.step_count % 100 == 0:
                self.weights_ref = self.learner.get_weights.remote()

            # Noisy Nets exploration: reset noise at each step
            self.model.reset_noise()

            # 3. Act
            with torch.inference_mode():
                obs_tensor = torch.as_tensor(
                    self.obs[None, ...], dtype=torch.float32, device=self.device
                )
                q_logits = self.model(obs_tensor)
                expected_qs = expected_value(q_logits, support=self.support)
                actions, _ = argmax_selector(expected_qs)
                action = actions.item()

            # 4. Step Env
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.episode_return += reward
            # TODO: URGENT. Creating a new tensor every step in the hotloop. should do in a more efficient way, maybe some thing like the rollout buffer in PPO (possible reuse?) ie store in a rollout buffer before sending to main replay buffer. Idea being its pre allocated basically. Must consider the N-Step case.
            n_step_transitions = self.n_step_proc(
                self.obs, action, reward, next_obs, terminated, truncated
            )

            # 5. Accumulate transitions locally
            for n_step_td in n_step_transitions:
                self.local_batch.append(n_step_td)

            self.obs = next_obs
            self.step_count += 1
            if terminated or truncated:
                wandb.log(
                    {
                        f"actor_{self.actor_id}/episode_return": self.episode_return,
                        "global/episode_return": self.episode_return,
                    },
                    step=self.step_count,
                )
                self.obs, _ = self.env.reset()
                self.episode_return = 0.0
                self.n_step_reset()

            # 6. Batch Priority Calculation and Push to remote buffer
            if len(self.local_batch) >= ACTOR_BATCH_SIZE:
                # Collate into a single batch
                collated = {
                    k: torch.cat([t[k] for t in self.local_batch]).to(self.device)
                    for k in self.local_batch[0].keys()
                }

                # Compute Initial Priorities in one batched forward pass
                with torch.no_grad():
                    q_logits = self.model(collated["obs"])
                    next_q_logits_online = self.model(collated["next_obs"])
                    next_q_logits_target = self.target_model(collated["next_obs"])

                    next_expected_qs = expected_value(next_q_logits_online, support=self.support)
                    next_actions, _ = argmax_selector(next_expected_qs)

                    td_target = self.target_fn(
                        next_q_logits_target,
                        next_actions.squeeze(-1),
                        collated["reward"],
                        collated["terminated"],
                        collated["gamma"],
                    )
                    pred_sa_logits = gather_q_values(q_logits, collated["action"])
                    _, info_dict = self.loss_fn(pred_sa_logits, td_target)

                # Attach priorities and move to CPU before sending to Ray
                collated["priority"] = info_dict["priorities"].unsqueeze(-1).cpu()

                # Move other keys to CPU as well
                push_batch = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in collated.items()
                }

                self.buffer.add_transitions.remote(push_batch)
                self.local_batch = []


@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class LearnerActor:
    def __init__(
        self,
        model_creator: Callable[[float], nn.Module],
        target_fn: Callable,
        loss_fn: Callable,
        buffer: ray.actor.ActorHandle,
        support: torch.Tensor,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.support = support.to(self.device)

        # Learner uses default sigma_init=0.5
        self.model = model_creator(0.5).to(self.device)
        self.target_model = model_creator(0.5).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.model.train()
        self.target_model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.buffer = buffer
        self.target_fn = partial(
            target_fn,
            support=self.support,
            v_min=V_MIN,
            v_max=V_MAX,
            atom_size=ATOM_SIZE,
        )
        self.loss_fn = loss_fn
        self.step_count = 0
        self.prefetch_queue = []

        wandb.init(
            project="rainbow-ape-x-ray-cartpole",
            group="ape-x-distributed",
            job_type="learner",
            config={
                "batch_size": BATCH_SIZE,
                "num_actors": NUM_ACTORS,
                "learning_rate": LEARNING_RATE,
                "noisy_nets": True,
            },
            mode="online",
        )
        print(f"Learner initialized on {self.device}.")

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def step(self) -> Dict[str, Any]:
        # 1. Prime/maintain the prefetch queue without blocking
        while len(self.prefetch_queue) < PREFETCH_COUNT:
            self.prefetch_queue.append(
                self.buffer.sample_batch.remote(BATCH_SIZE, BETA, MIN_BUFFER_SIZE)
            )

        # 2. Wait for at least one batch to be ready
        ready_refs, self.prefetch_queue = ray.wait(self.prefetch_queue, num_returns=1)
        result = ray.get(ready_refs[0])

        # 3. Handle the "buffer not ready" case
        if result[0] is None:
            # Maintain queue depth and return None to signal main to sleep
            self.prefetch_queue.append(
                self.buffer.sample_batch.remote(BATCH_SIZE, BETA, MIN_BUFFER_SIZE)
            )
            return None

        batch, indices, is_weights = result
        batch = {k: v.to(self.device) for k, v in batch.items()}
        indices = indices.to(self.device)
        is_weights = is_weights.to(self.device)

        # Reset noise for training step
        self.model.reset_noise()
        self.target_model.reset_noise()

        # 1. Forward Passes (Online and Target)
        q_logits = self.model(batch["obs"])
        with torch.no_grad():
            next_q_logits_online = self.model(batch["next_obs"])
            next_q_logits_target = self.target_model(batch["next_obs"])

            # 2. Next Action Selection (Double DQN: Online model selects)
            next_expected_qs = expected_value(next_q_logits_online, support=self.support)
            next_actions, _ = argmax_selector(next_expected_qs)

            # 3. Target Calculation
            td_target = self.target_fn(
                next_q_logits_target,
                next_actions.squeeze(-1),
                batch["reward"],
                batch["terminated"],
                batch["gamma"],
            )

        # 4. Prediction Extraction
        pred_sa_logits = gather_q_values(q_logits, batch["action"])

        # 5. Loss Calculation
        raw_loss, info = self.loss_fn(pred_sa_logits, td_target)
        weighted_loss = (raw_loss * is_weights).mean()

        # Augment info for logging
        with torch.no_grad():
            pred_sa_values = expected_value(
                pred_sa_logits.unsqueeze(1), self.support
            ).squeeze(1)
            info.update(
                {
                    "q_values/mean": pred_sa_values.mean().detach(),
                }
            )
        self.optimizer = apply_gradients(
            self.optimizer,
            weighted_loss,
            model=self.model,
            clip_grad_norm=MAX_GRAD_NORM,
        )
        td_errors = info["priorities"].detach().cpu()
        self.buffer.update_priorities.remote(indices.cpu(), td_errors, ALPHA)

        self.step_count += 1
        if self.step_count % TARGET_NET_UPDATE_FREQ == 0:
            hard_update_target_network(self.model, self.target_model)

        metrics = {
            "learner/loss": weighted_loss.item(),
            "learner/mean_q": info["q_values/mean"],
            # NOTE: buffer_size is no longer tracked here to avoid blocking calls
        }
        wandb.log(metrics, step=self.step_count)
        return metrics


def main():
    print("Initializing Ray...")
    ray.init()
    print("Ray initialized.")

    temp_env = gym.make(ENV_NAME)
    obs_shape = temp_env.observation_space.shape
    num_actions = temp_env.action_space.n
    temp_env.reset()
    temp_env.close()

    support = torch.linspace(V_MIN, V_MAX, ATOM_SIZE)

    # model_creator now takes sigma_init
    my_model_creator = lambda sigma: RainbowDQN(
        obs_shape, num_actions, ATOM_SIZE, support, sigma_init=sigma
    )

    my_target_fn = compute_categorical_q_td_target
    my_loss_fn = cross_entropy_loss

    print("Creating Replay Buffer...")
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
    buffer = ReplayBufferActor.remote(BUFFER_CAPACITY, buffer_shapes)

    print("Creating Learner...")
    learner = LearnerActor.remote(
        model_creator=my_model_creator,
        target_fn=my_target_fn,
        loss_fn=my_loss_fn,
        buffer=buffer,
        support=support,
    )

    print(f"Creating {NUM_ACTORS} Actors...")
    actors = [
        ActorActor.remote(
            actor_id=i,
            num_actors=NUM_ACTORS,
            model_creator=my_model_creator,
            target_fn=my_target_fn,
            loss_fn=my_loss_fn,
            learner=learner,
            buffer=buffer,
            support=support,
        )
        for i in range(NUM_ACTORS)
    ]

    print("Starting Actors...")
    for actor in actors:
        actor.run.remote()

    print("Starting Training Loop...")
    try:
        for step in range(MAX_LEARNER_STEPS):
            metrics = ray.get(learner.step.remote())
            if metrics and step % 100 == 0:
                print(f"Step {step}: Loss={metrics['learner/loss']:.4f}")
            elif not metrics:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
