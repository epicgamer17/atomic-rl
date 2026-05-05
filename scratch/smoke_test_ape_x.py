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

from functional.buffer import (
    init_per_buffer,
    sample_per,
    update_priorities,
    with_per_tracking,
    circular_write_strategy,
)
from functional.losses import bellman_error, mse_loss
from functional.targets import standard_td_target
from functional.action_selection import (
    double_selector,
    scalar_extractor,
    get_ape_x_epsilon,
)
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network

# --- Constants ---
ENV_NAME = "CartPole-v1"
SEED = 42

BATCH_SIZE = 512
GAMMA = 0.99
LEARNING_RATE = 1e-3
BUFFER_CAPACITY = 100_000
MIN_BUFFER_SIZE = 2000
TARGET_NET_UPDATE_FREQ = 1000
MAX_LEARNER_STEPS = 10

NUM_ACTORS = 4
ACTOR_BATCH_SIZE = 100

ALPHA = 0.6
BETA = 0.4

HIDDEN_SIZE = 512

# Set seeds for all backends
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DQN(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int):
        super().__init__()
        self.l1 = nn.Linear(input_shape[0], HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.l3 = nn.Linear(HIDDEN_SIZE, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity: int, shapes: Dict[str, Any]):
        self.buffer_state = init_per_buffer(capacity, shapes)
        # Tracking logic initialized once locally
        self.per_add = with_per_tracking(circular_write_strategy)

    def add_transitions(self, transitions_dict: Dict[str, torch.Tensor]):
        self.buffer_state = self.per_add(self.buffer_state, transitions_dict)

    def sample_batch(
        self, batch_size: int, beta: float
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        beta_tensor = torch.tensor(beta, dtype=torch.float32)
        return sample_per(self.buffer_state, batch_size, beta_tensor)

    def update_priorities(
        self, tree_indices: torch.Tensor, td_errors: torch.Tensor, alpha: float
    ):
        self.buffer_state = update_priorities(
            self.buffer_state, tree_indices, td_errors, alpha
        )

    def get_info(self) -> Dict[str, Any]:
        return {"size": self.buffer_state.size}


@ray.remote
class ActorActor:
    def __init__(
        self,
        actor_id: int,
        num_actors: int,
        model_creator: Callable[[], nn.Module],
        selector_fn: Callable,
        target_fn: Callable,
        loss_fn: Callable,
        learner: ray.actor.ActorHandle,
        buffer: ray.actor.ActorHandle,
    ):
        self.actor_id = actor_id
        self.env = gym.make(ENV_NAME)

        # Unique exploration rate per actor to ensure diverse experience
        self.epsilon = get_ape_x_epsilon(actor_id, num_actors)

        self.learner = learner
        self.buffer = buffer

        # Instantiate models dynamically from the injected creator
        self.model = model_creator()
        self.target_model = model_creator()
        self.num_actions = self.env.action_space.n

        # Store functional dependencies
        self.selector_fn = selector_fn
        self.target_fn = target_fn
        self.loss_fn = loss_fn

        self.local_batch = []
        self.obs, _ = self.env.reset(seed=SEED + actor_id)
        self.episode_return = 0.0
        self.step_count = 0

        # Initial weight fetch
        self.weights_ref = self.learner.get_weights.remote()

        # Initialize W&B locally on the actor.
        # We use a group name to tie them all together in the W&B UI.
        wandb.init(
            project="ape-x-ray-cartpole",
            group="ape-x-distributed",
            job_type=f"actor_{actor_id}",
            config={"actor_id": actor_id, "epsilon": self.epsilon},
        )

    def run(self):
        while True:
            # 1. Non-blocking weight sync
            ready, _ = ray.wait([self.weights_ref], timeout=0)
            if ready:
                weights = ray.get(self.weights_ref)
                self.model.load_state_dict(weights)
                self.target_model.load_state_dict(weights)
                self.weights_ref = self.learner.get_weights.remote()

            # 2. Act
            with torch.inference_mode():
                obs_tensor = torch.from_numpy(self.obs).float().unsqueeze(0)

                # Manual epsilon-greedy using the injected selector
                _, greedy_actions = self.selector_fn(self.model, None, obs_tensor)

                if random.random() < self.epsilon:
                    action = random.randint(0, self.num_actions - 1)
                    action_tensor = torch.tensor([[action]], dtype=torch.long)
                else:
                    action_tensor = greedy_actions
                    action = action_tensor.item()

            # 3. Step Env
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.episode_return += reward

            # 4. Compute Initial Priority
            transition_td = {
                "obs": obs_tensor,
                "action": action_tensor,
                "reward": torch.tensor([[reward]], dtype=torch.float32),
                "terminated": torch.tensor([[terminated]], dtype=torch.float32),
                "next_obs": torch.from_numpy(next_obs).float().unsqueeze(0),
                "gamma": torch.tensor([[GAMMA]], dtype=torch.float32),
            }

            with torch.no_grad():
                # Direct call to bellman_error without a wrapper function
                _, info_dict = bellman_error(
                    self.model,
                    self.target_model,
                    transition_td,
                    self.selector_fn,
                    partial(self.target_fn, gamma=transition_td["gamma"]),
                    loss_fn=self.loss_fn,
                )
            priority = info_dict["priorities"]

            # 5. Buffer transition
            self.local_batch.append(
                {
                    "obs": self.obs,
                    "action": action_tensor.reshape(1),
                    "reward": torch.tensor([reward], dtype=torch.float32),
                    "terminated": torch.tensor([terminated], dtype=torch.float32),
                    "truncated": torch.tensor([truncated], dtype=torch.float32),
                    "next_obs": next_obs,
                    "gamma": torch.tensor([GAMMA], dtype=torch.float32),
                    "priority": priority.reshape(1),
                }
            )

            self.obs = next_obs
            self.step_count += 1

            # Handle Episode End
            if terminated or truncated:
                # Log directly from the Actor.
                # Remove the self.learner.report_actor_metrics call!
                wandb.log(
                    {
                        f"actor_{self.actor_id}/episode_return": self.episode_return,
                        "global/episode_return": self.episode_return,  # W&B will aggregate this automatically
                    },
                    step=self.step_count,
                )
                self.obs, _ = self.env.reset()
                self.episode_return = 0.0

            # 6. Push to remote buffer
            if len(self.local_batch) >= ACTOR_BATCH_SIZE:
                collated = {
                    k: torch.stack([torch.as_tensor(t[k]) for t in self.local_batch])
                    for k in self.local_batch[0].keys()
                }
                self.buffer.add_transitions.remote(collated)
                self.local_batch = []


@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class LearnerActor:
    def __init__(
        self,
        model_creator: Callable[[], nn.Module],
        selector_fn: Callable,
        target_fn: Callable,
        loss_fn: Callable,
        buffer: ray.actor.ActorHandle,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model_creator().to(self.device)
        self.target_model = model_creator().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.buffer = buffer

        # Store functional dependencies
        self.selector_fn = selector_fn
        self.target_fn = target_fn
        self.loss_fn = loss_fn

        self.step_count = 0

        wandb.init(
            project="ape-x-ray-cartpole",
            group="ape-x-distributed",
            job_type="learner",
            config={
                "batch_size": BATCH_SIZE,
                "num_actors": NUM_ACTORS,
                "learning_rate": LEARNING_RATE,
                "gamma": GAMMA,
                "alpha": ALPHA,
                "beta": BETA,
            },
        )

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def step(self) -> Dict[str, Any]:
        buffer_info = ray.get(self.buffer.get_info.remote())
        if buffer_info["size"] < MIN_BUFFER_SIZE:
            return None

        batch, indices, is_weights = ray.get(
            self.buffer.sample_batch.remote(BATCH_SIZE, BETA)
        )

        batch = {k: v.to(self.device) for k, v in batch.items()}
        indices = indices.to(self.device)
        is_weights = is_weights.to(self.device)

        # Calculate Loss using injected functions
        loss, info = bellman_error(
            self.model,
            self.target_model,
            batch,
            self.selector_fn,
            partial(self.target_fn, gamma=batch["gamma"]),
            loss_fn=self.loss_fn,
        )

        weighted_loss = (loss * is_weights).mean()
        self.optimizer = apply_gradients(self.optimizer, weighted_loss)

        # Update Priorities
        td_errors = info["priorities"].detach().cpu()
        self.buffer.update_priorities.remote(indices.cpu(), td_errors, ALPHA)

        self.step_count += 1
        if self.step_count % TARGET_NET_UPDATE_FREQ == 0:
            hard_update_target_network(self.model, self.target_model)

        metrics = {
            "learner/loss": weighted_loss.item(),
            "learner/mean_q": info["q_values/mean"],
            "buffer_size": buffer_info["size"],
        }
        wandb.log(metrics, step=self.step_count)
        return metrics


def main():
    ray.init()

    # 1. Warm up environment to suppress gymnasium registry warnings
    temp_env = gym.make(ENV_NAME)
    obs_shape = temp_env.observation_space.shape
    num_actions = temp_env.action_space.n
    temp_env.reset()
    temp_env.close()

    # 2. Define the Algorithm Components (Transparent and Reusable)
    my_model_creator = lambda: DQN(obs_shape, num_actions)
    my_selector_fn = partial(double_selector, extractor_fn=scalar_extractor)
    my_target_fn = standard_td_target
    my_loss_fn = mse_loss

    # 3. Initialize Buffer
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

    # 4. Initialize Learner with Functional Logic
    learner = LearnerActor.remote(
        model_creator=my_model_creator,
        selector_fn=my_selector_fn,
        target_fn=my_target_fn,
        loss_fn=my_loss_fn,
        buffer=buffer,
    )

    # 5. Initialize Actors with Functional Logic
    actors = [
        ActorActor.remote(
            actor_id=i,
            num_actors=NUM_ACTORS,
            model_creator=my_model_creator,
            selector_fn=my_selector_fn,
            target_fn=my_target_fn,
            loss_fn=my_loss_fn,
            learner=learner,
            buffer=buffer,
        )
        for i in range(NUM_ACTORS)
    ]

    for actor in actors:
        actor.run.remote()

    print("Starting training loop...")
    try:
        for step in range(MAX_LEARNER_STEPS):
            metrics = ray.get(learner.step.remote())
            if metrics and step % 100 == 0:
                print(
                    f"Step {step}: Loss={metrics['learner/loss']:.4f}, Buffer={metrics['buffer_size']}"
                )
            elif not metrics:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
