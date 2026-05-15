# Fully Generated
"""
Notes on PPO for Pendulum (Continuous Action Space):

This example demonstrates Proximal Policy Optimization (PPO) applied to the Pendulum-v1 environment.
Pendulum-v1 features a 3D observation space (cos(theta), sin(theta), theta_dot) and a 1D continuous
action space (torque) in the range [-2.0, 2.0].

Key Implementation Details:
1.  **Explicit Continuous Architecture**: Uses independent actor and critic networks.
    The actor predicts the mean of a Gaussian distribution, while the log standard deviation
    is a standalone learnable parameter.
2.  **Action Selection**: Employs `sample_distribution` to handle continuous action sampling
     and log-probability calculation.
3.  **Normalization**: Uses `NormalizeObservation` and `NormalizeReward` wrappers, which are
    typically critical for stable training in continuous control tasks.
4.  **GAE & PPO Loss**: Implements Generalized Advantage Estimation (GAE) and the clipped surrogate
    objective as per Schulman et al. (2017).
"""

# TODO: not working, i know it worked before, i think when i was using pufferlib, but strangely pufferlib stopped cartpole from working well.
# TODO: attempt a cleanup if possible
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb
from einops import rearrange

from functional.action_selection import sample_distribution
from functional.optimizer import apply_gradients
from functional.returns import compute_gae
from functional.losses import (
    clipped_surrogate_loss,
    entropy_loss,
    probability_ratio,
    clipped_mse_loss,
    mse_loss,
)
from torch.optim.lr_scheduler import LinearLR
from functional.visualization import compute_explained_variance
from functional.network import layer_init
from functional.rollout_buffer import (
    init_rollout_buffer,
    store_rollout_step,
    flatten_rollout_buffer,
    record_truncations,
    get_rollout_next_values,
    yield_shuffled_minibatches,
)
from functional.utils import (
    ema_update,
    standardize_tensor,
    set_seed,
    to_tensor,
    to_numpy_action,
)
from tensordict import TensorDict
from envs.wrappers import VecNormalize

# Constants
LEARNING_RATE = 3e-4
MAX_ITERATIONS = 500  # Sufficient for Pendulum
GAMMA = 0.9
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.0
CRITIC_COEFF = 0.5
MAX_GRAD_NORM = 0.5
STEPS_PER_ENV = 128  # One full Pendulum episode
UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 64
CLIP_COEF = 0.2
TARGET_KL = 0.015
NUM_ENVS = 4
SEED = 42

# Seeding for reproducibility
set_seed(SEED)


class ActorCritic(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        # 1. Critic Network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # 2. Actor Mean Network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
        )

        # 3. Actor LogStd (Independent, learnable parameter)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

    def forward(self, x):
        """
        Forward pass for the actor-critic network.

        Args:
            x (torch.Tensor): Observation input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: action_mean, action_std, values
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        values = self.critic(x)
        return action_mean, action_std, values


# --- 1. Initialization ---
def make_env(env_id, seed, idx):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Pendulum actions are in [-2, 2], ClipAction ensures they stay within bounds
        env = gym.wrappers.ClipAction(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# TODO: switch to either procgen, envpool, isaac gym, brax, or pufferlib to decrease training time (same number iterations but faster)
# Initialize vectorized environments
envs = gym.vector.SyncVectorEnv(
    [make_env("Pendulum-v1", SEED + i, i) for i in range(NUM_ENVS)]
)
envs = VecNormalize(
    envs,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=GAMMA,
)


obs_shape = envs.single_observation_space.shape
num_actions = envs.single_action_space.shape[0]
device = torch.device("cpu")

model = ActorCritic(obs_shape, num_actions).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

scheduler = LinearLR(
    optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERATIONS
)

obs, info = envs.reset(seed=SEED)

# Pre-allocate rollout buffers
shapes = {
    "observations": obs_shape,
    "actions": (1,),
    "logprobs": (1,),
    "rewards": (),
    "terminated": (),
    "truncated": (),
    "values": (1,),
    "means": (num_actions,),
    "stds": (num_actions,),
}
buffer = init_rollout_buffer(
    steps_per_env=STEPS_PER_ENV,
    num_envs=NUM_ENVS,
    shapes=shapes,
    device=device,
)

# Pre-allocate rollout buffers

rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

# Initialize W&B
wandb.init(
    project="ppo-pendulum",
    config={
        "lr": LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "update_epochs": UPDATE_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "clip_coef": CLIP_COEF,
        "num_envs": NUM_ENVS,
        "steps_per_env": STEPS_PER_ENV,
        "env": "Pendulum-v1",
    },
)
wandb.define_metric("*", step_metric="global_step")
global_step = 0

# Training Loop
for iteration in range(MAX_ITERATIONS):
    # 1. Data Collection Phase
    with torch.inference_mode():
        for step in range(STEPS_PER_ENV):
            obs_tensor = to_tensor(obs, device=device)
            action_mean, action_std, value = model(obs_tensor)

            # Use sample_distribution for continuous actions
            dist = torch.distributions.Normal(action_mean, action_std)
            action, info_dict = sample_distribution(dist, explore=True)

            # Step Env expects numpy arrays for actions
            action_np = to_numpy_action(action)

            # Step Env
            next_obs, reward, terminated, truncated, info = envs.step(action_np)
            global_step += NUM_ENVS

            # Store in rollout buffer
            transition = TensorDict(
                {
                    "observations": obs_tensor,
                    "actions": action,
                    "logprobs": info_dict["log_prob"]
                    .sum(dim=-1, keepdim=True)
                    .detach(),
                    "rewards": to_tensor(reward, device=device),
                    "terminated": to_tensor(terminated, device=device),
                    "truncated": to_tensor(truncated, device=device),
                    "values": value.detach(),
                    "means": action_mean.detach(),
                    "stds": action_std.detach(),
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
                        torch.as_tensor(
                            env_indices[trunc_mask], dtype=torch.long, device=device
                        ),
                        torch.as_tensor(
                            final_obs[trunc_mask], dtype=torch.float32, device=device
                        ),
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

        # Compute last values for GAE
        last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        _, _, last_values = model(last_obs_tensor)

        next_values = get_rollout_next_values(
            buffer, last_values, get_value_fn=lambda obs: model(obs)[2], device=device
        )

    # 2. Advantage & Target Calculation
    advantages = compute_gae(
        rewards=buffer.data["rewards"],
        terminated=buffer.data["terminated"],
        truncated=buffer.data["truncated"],
        values=buffer.data["values"].squeeze(-1),
        next_values=next_values.squeeze(-1),
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
    )

    # Explicit mathematical derivation (Explicit over implicit, no redundant compute)
    returns = advantages.unsqueeze(-1) + buffer.data["values"]

    # Flatten buffer for training
    flat_data = flatten_rollout_buffer(buffer)
    flat_advantages = rearrange(advantages, "b t -> (b t) 1")
    flat_returns = rearrange(returns, "b t 1 -> (b t) 1")

    flat_data["advantages"] = flat_advantages
    flat_data["returns"] = flat_returns

    # 3. Optimization Phase
    epoch_losses = []
    clip_fractions = []
    approx_kls = []

    for epoch in range(UPDATE_EPOCHS):
        epoch_kls = []
        for mb in yield_shuffled_minibatches(
            flat_data, MINIBATCH_SIZE, generator=rng_key
        ):
            # Re-evaluate model on minibatch
            new_means, new_stds, new_values = model(mb["observations"])

            # Re-create normal distribution for new log probabilities
            dist = torch.distributions.Normal(new_means, new_stds)
            dist = torch.distributions.Independent(dist, 1)
            # Log prob for continuous actions [B, num_actions] -> [B]
            new_log_probs = dist.log_prob(mb["actions"])

            # 1. Policy Loss (Clipped Surrogate)
            ratio = probability_ratio(
                old_log_probs=mb["logprobs"],
                new_log_probs=new_log_probs,
            )

            mb_advantages = mb["advantages"]
            mb_advantages = standardize_tensor(mb_advantages)

            pg_loss, pg_info = clipped_surrogate_loss(
                ratio=ratio,
                advantages=mb_advantages,
                clip_coef=CLIP_COEF,
            )
            pg_loss = pg_loss.mean()

            # 2. Value Loss
            critic_loss, _ = clipped_mse_loss(
                predictions=new_values,
                old_predictions=mb["values"],
                targets=mb["returns"],
                clip_coef=CLIP_COEF,
            )
            critic_loss = critic_loss.mean()

            # 3. Entropy Loss
            ent_loss, _ = entropy_loss(dist)
            ent_loss = ent_loss.mean()

            # Total Loss
            loss = pg_loss + CRITIC_COEFF * critic_loss + ENTROPY_COEFF * ent_loss

            # Backprop
            optimizer = apply_gradients(
                optimizer, loss, model=model, clip_grad_norm=MAX_GRAD_NORM
            )

            # Metrics tracking
            with torch.no_grad():
                epoch_kls.append(pg_info["policy/approx_kl"].item())
                clip_fractions.append(pg_info["policy/clip_fraction"].item())
                approx_kls.append(pg_info["policy/approx_kl"].item())
                epoch_losses.append(loss.item())

        # Early stopping based on KL
        if TARGET_KL is not None and np.mean(epoch_kls) > 1.5 * TARGET_KL:
            break

    # Step the learning rate down
    scheduler.step()
    buffer.truncation_records.clear()

    # Logging
    if iteration % 10 == 0:
        explained_var = compute_explained_variance(
            flat_returns.detach().cpu().numpy(),
            flat_data["values"].detach().cpu().numpy(),
        )

        log_dict = {
            "learning_rate": scheduler.get_last_lr()[0],
            "loss/total": np.mean(epoch_losses),
            "loss/critic": critic_loss.item(),
            "loss/policy": pg_loss.item(),
            "loss/entropy": ent_loss.item(),
            "value/mean": flat_data["values"].mean().item(),
            "value/return_mean": flat_returns.mean().item(),
            "value/explained_variance": explained_var,
            "advantages/mean": flat_advantages.mean().item(),
            "advantages/std": flat_advantages.std().item(),
            "ppo/clip_fraction": np.mean(clip_fractions),
            "ppo/approx_kl": np.mean(approx_kls),
            "global_step": global_step,
        }
        wandb.log(log_dict)

wandb.finish()
