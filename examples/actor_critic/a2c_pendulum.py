"""
Notes on A2C for Pendulum (Continuous Action Space):

STRUCTURALLY SIMILAR TO a2c_cartpole.py but adapted for continuous actions.
Uses standard Gymnasium Vector Envs and the functional rollout buffer system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb
from einops import rearrange
from functools import partial

from functional.action_selection import gaussian_sampling_selector
from functional.optimizer import apply_gradients
from functional.returns import compute_n_step_returns
from functional.losses import policy_gradient_loss, mse_loss, entropy_loss
from torch.optim.lr_scheduler import LinearLR
from functional.visualization import compute_explained_variance
from functional.network import layer_init
from functional.rollout_buffer import (
    init_rollout_buffer,
    store_rollout_step,
    flatten_rollout_buffer,
    record_truncations,
    get_rollout_next_values,
)
from functional.utils import standardize_tensor
from tensordict import TensorDict

# Constants
LEARNING_RATE = 1e-4
MAX_ITERATIONS = 100_000
GAMMA = 0.9
CRITIC_COEFF = 0.5
ENTROPY_COEFF = 0.0
MAX_GRAD_NORM = 0.5
N_STEP = 5
STEPS_PER_ENV = N_STEP
NUM_ENVS = 32
SEED = 42
MAX_ACTION = 2.0
HIDDEN_SIZE = 64
INITIAL_LOG_STD = 0.0

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Decoupled networks for Actor-Critic
class ActorCritic(nn.Module):
    """
    Actor-Critic network for continuous action spaces with decoupled networks.
    This architecture uses separate parameters for the actor and critic to
    handle different gradient magnitudes, which is often necessary for
    stability in continuous domains like Pendulum.
    """

    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        # NOTE: Details from the A3C paper for continuous control:
        # 1. State-dependent variance: Both mu and log_std are network heads. This is recommended but we do not implement it here as we find it leads to instability.
        # 2. Separate Networks: The actor and critic networks are completely
        #    separate (not sharing layers) to prevent interference between updates.

        # Actor Network: Predicts distribution parameters (mu and log_std)
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.Tanh(),
        )
        self.actor_mu = layer_init(nn.Linear(HIDDEN_SIZE, num_actions), std=0.01)
        self.log_std = nn.Parameter(torch.ones(1, num_actions) * INITIAL_LOG_STD)

        # Critic Network: Predicts the state value estimate
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.actor_backbone(x)
        mu = torch.tanh(self.actor_mu(latent)) * MAX_ACTION
        std = torch.exp(self.log_std).expand_as(mu)
        value = self.critic(x)
        return mu, std, value


# --- 1. Initialization (Defining the State) ---
def make_env(env_id, seed, idx):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# Initialize vectorized environments using standard Gymnasium Vector Envs
# TODO: switch to either procgen, envpool, isaac gym, brax, or pufferlib to decrease training time (same number iterations but faster)
envs = gym.vector.SyncVectorEnv(
    [make_env("Pendulum-v1", SEED + i, i) for i in range(NUM_ENVS)]
)

obs_shape = envs.single_observation_space.shape
num_actions = envs.single_action_space.shape[0]
device = torch.device("cpu")

model = ActorCritic(obs_shape, num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

# Linearly decay LR from 1.0 * LEARNING_RATE to 0.0 * LEARNING_RATE over MAX_ITERATIONS
scheduler = LinearLR(
    optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERATIONS
)

obs, info = envs.reset(seed=SEED)

# Pre-allocate rollout buffers
shapes = {
    "observations": obs_shape,
    "actions": (num_actions,),
    "logprobs": (),
    "rewards": (),
    "terminated": (),
    "truncated": (),
    "values": (),
    "mu": (num_actions,),
    "std": (num_actions,),
}
buffer = init_rollout_buffer(
    steps_per_env=STEPS_PER_ENV,
    num_envs=NUM_ENVS,
    shapes=shapes,
    device=device,
)

# Pre-allocate rollout buffers

# Initialize W&B
wandb.init(project="a2c-pendulum", config={"lr": LEARNING_RATE, "gamma": GAMMA})
wandb.define_metric("*", step_metric="global_step")
global_step = 0

for iteration in range(MAX_ITERATIONS):
    # NOTE: vectorized collection with inference_mode
    with torch.inference_mode():
        for step in range(STEPS_PER_ENV):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mu, std, value = model(obs_tensor)

            action_tensor, info_dict = gaussian_sampling_selector(mu, std, explore=True)
            action_np = action_tensor.cpu().numpy()
            # Pendulum actions are already scaled by mu head, but we can clip to be safe
            action_np = np.clip(action_np, -MAX_ACTION, MAX_ACTION)

            # 2. Step Env
            next_obs, reward, terminated, truncated, info = envs.step(action_np)
            global_step += NUM_ENVS

            # 3. Add to buffers
            transition = TensorDict(
                {
                    "observations": obs_tensor,
                    "actions": action_tensor,
                    "logprobs": info_dict["log_prob"].squeeze(-1).detach(),
                    "rewards": torch.as_tensor(
                        reward, dtype=torch.float32, device=device
                    ),
                    "terminated": torch.as_tensor(
                        terminated, dtype=torch.float32, device=device
                    ),
                    "truncated": torch.as_tensor(
                        truncated, dtype=torch.float32, device=device
                    ),
                    "values": value.squeeze(-1).detach(),
                    "mu": mu.detach(),
                    "std": std.detach(),
                },
                batch_size=[NUM_ENVS],
            )
            store_rollout_step(buffer=buffer, step=step, transition=transition)

            # 4. Handle Truncations (Gymnasium auto-resets)
            if "final_observation" in info:
                from functional.utils import extract_vector_env_final_obs

                env_indices, final_obs = extract_vector_env_final_obs(info)
                # Filter to only record environments that were truncated
                trunc_mask = truncated[env_indices]
                if trunc_mask.any():
                    record_truncations(
                        buffer,
                        step,
                        torch.as_tensor(env_indices[trunc_mask], dtype=torch.long, device=device),
                        torch.as_tensor(final_obs[trunc_mask], dtype=torch.float32, device=device),
                    )

            if "final_info" in info:
                for item in info["final_info"]:
                    if item is not None and "episode" in item:
                        wandb.log(
                            {
                                "episode_return": item["episode"]["r"][0],
                                "episode_length": item["episode"]["l"][0],
                                "global_step": global_step,
                            }
                        )

            obs = next_obs

        last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        _, _, last_values = model(last_obs_tensor)

        # Calculate Loss & Gradients
        next_values = get_rollout_next_values(
            buffer,
            last_values,
            get_value_fn=lambda obs: model(obs)[2],
            device=device,
        )

    # --- 3. The Update Loop ---

    returns = compute_n_step_returns(
        rewards=buffer.data["rewards"],
        terminated=buffer.data["terminated"],
        truncated=buffer.data["truncated"],
        values=buffer.data["values"],
        next_values=next_values,
        gamma=GAMMA,
        n=N_STEP,
    )

    # 2. Define Baseline & Calculate Raw Advantage (Explicit Math)
    baseline = buffer.data["values"].detach()
    advantages = returns - baseline

    # 3. Optional Scaling
    advantages = standardize_tensor(advantages)

    # Flatten buffer for loss calculations
    flat_data = flatten_rollout_buffer(buffer)
    flat_advantages = rearrange(advantages, "b t -> (b t)")
    flat_returns = rearrange(returns, "b t -> (b t)")

    # --- Re-evaluation Pass ---
    new_mu, new_std, new_values = model(flat_data["observations"])
    new_values = new_values.squeeze(-1)

    # Re-calculate log probabilities for the actions taken
    dist = torch.distributions.Normal(new_mu, new_std)
    new_log_probs = dist.log_prob(flat_data["actions"]).sum(dim=-1)

    pg_loss, info_dict = policy_gradient_loss(
        advantages=flat_advantages,
        log_probs=new_log_probs,
    )
    pg_loss = pg_loss.mean()

    ent_loss, _ = entropy_loss(dist)
    ent_loss = ent_loss.mean()

    critic_loss, _ = mse_loss(predictions=new_values, targets=flat_returns.detach())
    critic_loss = critic_loss.mean()

    loss = pg_loss + CRITIC_COEFF * critic_loss - ENTROPY_COEFF * ent_loss

    # Apply Updates
    optimizer = apply_gradients(
        optimizer, loss, model=model, clip_grad_norm=MAX_GRAD_NORM
    )

    # Step the learning rate down
    scheduler.step()

    # Clear truncation records for the next iteration
    buffer.truncation_records.clear()

    if iteration % 100 == 0:
        explained_var = compute_explained_variance(
            returns.detach().cpu().numpy(), buffer.data["values"].detach().cpu().numpy()
        )

        log_dict = info_dict.copy()
        log_dict.update(
            {
                "learning_rate": scheduler.get_last_lr()[0],  # Log the decaying LR
                "loss/total": loss.item(),
                "loss/critic": critic_loss.item(),
                "value/mean": buffer.data["values"].mean().item(),
                "value/return_mean": returns.mean().item(),
                "value/explained_variance": explained_var,
                "advantages/mean": advantages.mean().item(),
                "std/mean": buffer.data["std"].mean().item(),
                "global_step": global_step,
            }
        )
        wandb.log(log_dict)

wandb.finish()
