"""
Notes on PPO:

The PPO paper argues that there is room for improvement in developing a method that is scalable (to large models and parallel implementations), data efficient, and robust (i.e., successful on a variety of problems without hyperparameter tuning).

Q-learning (with function approximation) fails on many simple problems (continuous control) and is poorly understood, vanilla policy gradient methods have poor data effiency and robustness; and trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks).

PPO aims to achieve the data efficiency and reliable performance of TRPO, but with the implementation simplicity of a standard gradient method.

It does this by taking multiple gradient steps on the same data (typically 10-20, compared to 1 for vanilla policy gradient and TRPO), but with a constraint that prevents it from deviating too far from the previous policy.

The paper states that while it is appealing to perform multiple steps of optimization on the standard policy gradient loss using the same trajectory, doing so is not well-justified, and empirically it often leads to destructively large policy updates.

It introduces a clipped surrogate objective function to get a lower bound on the performance of the policy and a update pattern where they sample data from the policy and then perform several epochs of updates on that data (as opposed to just one update per sample like in vanilla policy gradient, or doing updates on the fly like in A2C). The motivation is that L^CLI which L^CLIP (the PPO clipped surrogate loss) is based on, doesn't penalize moving away from the old policy, so it leads to large policy updates that move the policy ratio away from 1 (ie it becomes more off policy preventing multiple epochs).

Another approach which can be used as an alternative to the clipped surrogate loss, or in addition to it is to add a penalty on have a target KL divergence. If the policy strays too far from the previous policy, it is penalized. The goal of this is to achieve a target kl divergence (change in policy) each policy update (after all epochs for a single update step). This was found to perform worse than the clipped surrogate loss in the papers experiments.

It performs better on continuous control tasks than A2C and vanilla policy gradient and has better sample complexity than A2C on Atari.

NOTE: the paper is a very easy read and it is recommend to read it
NOTE: we do not implement the KL divergence penalty version of PPO in this implementation, as it was largely abandoned by the community in favor of the clipped objective.

TODO: what is better sample complexity
TODO: More on pessemistic lower bound, what it means, why its useful, etc
"""

# TODO: attempt a cleanup if possible
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
from envs.wrappers import VecNormalizeObservation

# Constants
LEARNING_RATE = 2.5e-4
MAX_ITERATIONS = 976
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.01
CRITIC_COEFF = 0.5
MAX_GRAD_NORM = 0.5
STEPS_PER_ENV = 128
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 128
CLIP_COEF = 0.2
TARGET_KL = None
NUM_ENVS = 4
SEED = 42

# Seeding for reproducibility
set_seed(SEED)


# NOTE: PPO can have a shared backbone between the actor and critic or seperate networks.
class ActorCritic(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        x_actor = self.actor(x)
        x_critic = self.critic(x)
        return x_actor, x_critic


# --- 1. Initialization (Defining the State) ---
def make_env(env_id, seed, idx):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# Use standard Gymnasium Vector Envs instead of Pufferlib
# TODO: switch to either procgen, envpool, isaac gym, brax, or pufferlib to decrease training time (same number iterations but faster)
envs = gym.vector.SyncVectorEnv(
    [make_env("CartPole-v1", SEED + i, i) for i in range(NUM_ENVS)]
)

# NOTE: PPO on Cartpole at least seems to be highly sensitive to this.
# TODO: normalizing obs seems to break cartpole idk why
# envs = VecNormalizeObservation(envs, clip_obs=10.0, gamma=GAMMA)

obs_shape = envs.single_observation_space.shape
num_actions = envs.single_action_space.n
device = torch.device("cpu")

model = ActorCritic(obs_shape, num_actions).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

scheduler = LinearLR(
    optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERATIONS
)

obs, info = envs.reset(seed=SEED)

# Pre-allocate rollout buffers using the new functional system
shapes = {
    "observations": obs_shape,
    "actions": (1,),
    "logprobs": (1,),
    "rewards": (),
    "terminated": (),
    "truncated": (),
    "values": (1,),
    "logits": (num_actions,),
}
buffer = init_rollout_buffer(
    steps_per_env=STEPS_PER_ENV,
    num_envs=NUM_ENVS,
    shapes=shapes,
    device=device,
)

# Pre-allocate rollout buffers using the new functional system

rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

# Initialize W&B
wandb.init(
    project="ppo-cartpole",
    config={
        "lr": LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "update_epochs": UPDATE_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "clip_coef": CLIP_COEF,
        "num_envs": NUM_ENVS,
        "steps_per_env": STEPS_PER_ENV,
    },
)
wandb.define_metric("*", step_metric="global_step")
global_step = 0

# Using full episodes
for iteration in range(MAX_ITERATIONS):
    # 1. Data Collection Phase
    with torch.inference_mode():
        for step in range(STEPS_PER_ENV):
            obs_tensor = to_tensor(obs, device=device)
            logits, value = model(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action, info_dict = sample_distribution(dist, explore=True)
            action_np = to_numpy_action(action)

            # 2. Step Env
            next_obs, reward, terminated, truncated, info = envs.step(action_np)
            global_step += NUM_ENVS

            # 3. Add to "online" buffers
            transition = TensorDict(
                {
                    "observations": obs_tensor,
                    "actions": action,
                    "logprobs": info_dict["log_prob"].detach(),
                    "rewards": to_tensor(reward, device=device),
                    "terminated": to_tensor(terminated, device=device),
                    "truncated": to_tensor(truncated, device=device),
                    "values": value.detach(),
                    "logits": logits.detach(),
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

            # NOTE: obs = next_obs is moved to here.
            obs = next_obs

        # Compute last values for the re-evaluation pass
        last_obs_tensor = to_tensor(obs, device=device)
        _, last_values = model(last_obs_tensor)

        # Calculate Next Values (handling truncations)
        next_values = get_rollout_next_values(
            buffer, last_values, get_value_fn=lambda obs: model(obs)[1], device=device
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

    # Flatten buffer for loss calculations
    flat_data = flatten_rollout_buffer(buffer)
    flat_advantages = rearrange(advantages, "b t -> (b t) 1")
    flat_returns = rearrange(returns, "b t 1 -> (b t) 1")

    # Add advantages and returns to flat_data for easier minibatch sampling
    flat_data["advantages"] = flat_advantages
    flat_data["returns"] = flat_returns

    # 3. The Update Loop (Multiple Epochs & Minibatches)
    epoch_losses = []
    clip_fractions = []
    approx_kls = []

    for epoch in range(UPDATE_EPOCHS):
        epoch_kls = []
        for mb in yield_shuffled_minibatches(
            flat_data, MINIBATCH_SIZE, generator=rng_key
        ):
            # Re-evaluate the policy and value function on the minibatch
            new_logits, new_values = model(mb["observations"])

            # Re-calculate log probabilities and entropy
            dist = torch.distributions.Categorical(logits=new_logits)
            new_log_probs = dist.log_prob(mb["actions"])

            # 1. Policy Loss (Clipped Surrogate)
            ratio = probability_ratio(
                old_log_probs=mb["logprobs"],
                new_log_probs=new_log_probs,
            )

            # Advantage Standardisation (Minibatch level as per CleanRL)
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

            # Apply Updates
            optimizer = apply_gradients(
                optimizer, loss, model=model, clip_grad_norm=MAX_GRAD_NORM
            )

            # Track Metrics
            with torch.no_grad():
                epoch_kls.append(pg_info["policy/approx_kl"].item())
                clip_fractions.append(pg_info["policy/clip_fraction"].item())
                approx_kls.append(pg_info["policy/approx_kl"].item())
                epoch_losses.append(loss.item())

        # KL early stop on the per-epoch mean (CleanRL-style): break the
        # epoch loop only, after the epoch's gradient steps have been taken.
        if TARGET_KL is not None and np.mean(epoch_kls) > 1.5 * TARGET_KL:
            break

    # Step the learning rate down
    scheduler.step()

    # Clear truncation records for the next iteration
    # TODO: do this here or start of collection phase?
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
