# Fully Generated
# TODO: compare with 37 implementation details of PPO results, mine seem lower/have less variance than theirs for sure and a mean of 1000 after 1M steps instead of 2000 like theirs.
# TODO: attempt a cleanup if possible
# TODO: notes on multi continuous
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
from functional.returns import compute_gae, compute_td_lambda_returns
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
from functional.utils import standardize_tensor
from tensordict import TensorDict

# Constants
LEARNING_RATE = 3e-4
MAX_ITERATIONS = 488  # ~2M steps for HalfCheetah
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.0
CRITIC_COEFF = 0.5
MAX_GRAD_NORM = 0.5
STEPS_PER_ENV = 2048
UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 64
CLIP_COEF = 0.2
TARGET_KL = None
NUM_ENVS = 1
SEED = 42

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10).astype(np.float32)
        )
        env = gym.wrappers.NormalizeReward(env, gamma=GAMMA)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# TODO: switch to either procgen, envpool, isaac gym, brax, or pufferlib to decrease training time (same number iterations but faster)
# Initialize vectorized environments using standard Gymnasium Vector Envs
envs = gym.vector.SyncVectorEnv(
    [make_env("HalfCheetah-v4", SEED + i, i) for i in range(NUM_ENVS)]
)


obs_shape = envs.single_observation_space.shape
num_actions = envs.single_action_space.shape[0]
device = torch.device("cpu")  # Using CPU for now, can be changed to "cuda" if available

model = ActorCritic(obs_shape, num_actions).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

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
    project="ppo-mujoco",
    config={
        "lr": LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "update_epochs": UPDATE_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "clip_coef": CLIP_COEF,
        "num_envs": NUM_ENVS,
        "steps_per_env": STEPS_PER_ENV,
        "env": "HalfCheetah-v4",
    },
)
wandb.define_metric("*", step_metric="global_step")
global_step = 0

# Training Loop
for iteration in range(MAX_ITERATIONS):
    # 1. Data Collection Phase
    with torch.inference_mode():
        for step in range(STEPS_PER_ENV):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action_mean, action_std, value = model(obs_tensor)

            # Use sample_distribution for continuous actions
            dist = torch.distributions.Normal(action_mean, action_std)
            action, info_dict = sample_distribution(dist, explore=True)

            # Step Env expects numpy arrays for actions
            action_np = action.cpu().numpy().astype(np.float32)

            # Step Env
            next_obs, reward, terminated, truncated, info = envs.step(action_np)
            global_step += NUM_ENVS

            # Store in rollout buffer
            transition = TensorDict(
                {
                    "observations": obs_tensor,
                    "actions": action,
                    "logprobs": info_dict["log_prob"].sum(dim=-1).detach(),
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
        last_values = last_values.squeeze(-1)

        next_values = get_rollout_next_values(
            buffer, last_values, get_value_fn=lambda obs: model(obs)[2], device=device
        )

    # 2. Advantage & Target Calculation
    advantages = compute_gae(
        rewards=buffer.data["rewards"],
        terminated=buffer.data["terminated"],
        truncated=buffer.data["truncated"],
        values=buffer.data["values"],
        next_values=next_values,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
    )

    returns = compute_td_lambda_returns(
        rewards=buffer.data["rewards"],
        terminated=buffer.data["terminated"],
        truncated=buffer.data["truncated"],
        values=buffer.data["values"],
        next_values=next_values,
        gamma=GAMMA,
        lam=GAE_LAMBDA,
    )

    # Flatten buffer for training
    flat_data = flatten_rollout_buffer(buffer)
    flat_advantages = rearrange(advantages, "b t -> (b t)")
    flat_returns = rearrange(returns, "b t -> (b t)")

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
            new_values = new_values.squeeze(-1)

            # Re-create normal distribution for new log probabilities
            dist = torch.distributions.Normal(new_means, new_stds)
            dist = torch.distributions.Independent(dist, 1)
            # Log prob for continuous actions [B, num_actions] -> [B] automatically via Independent
            new_log_probs = dist.log_prob(mb["actions"])

            # Compute KL divergence
            with torch.no_grad():
                log_ratio = new_log_probs - mb["logprobs"]
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
            epoch_kls.append(approx_kl)

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
                clipped = (ratio - 1.0).abs() > CLIP_COEF
                clip_fractions.append(clipped.float().mean().item())
                approx_kls.append(approx_kl)
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
            "ppo/actor_logstd": model.actor_logstd.mean().item(),
            "global_step": global_step,
        }
        wandb.log(log_dict)

wandb.finish()
