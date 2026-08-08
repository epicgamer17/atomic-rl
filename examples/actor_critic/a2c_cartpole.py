"""
Notes on A2C:

Essentially REINFORCE + a value function for the baseline to compute advantages. Or another way of putting it is REINFORCE with learned state dependant baseline.

Not exactly related to A2C, but argued in the A3C paper. To attempt to get the same benefits of Experience Replay without the memory footprint, we can use multiple workers (or a vectorized env) to collect data in parallel (asynchronously for A3C, synchronously for A2C). This helps to decorrelate the data similar to Experience Replay (since the samples come from different trajectories so they will have a different variety of states explored, even if they are from the same policy and environment). The data is online so it is discarded after each update step, allowing for a smaller memory footprint.

Specific to the A3C paper, the method they used was significantly more efficient than DQN while using a single machine and no GPU. They achieve better results than DQN and Gorilla (APE-X predecessor) with less wall clock time, less compute resources, and without a massive distributed system. Our implementation will use a vectorized synchronous environment and attempt to benefit from the GPU in the update step, the standard way to implement A2C on one machine.

The idea behind Advantage Actor-Critic is to reduce the variance of the policy gradient updates by using a baseline to compute advantages. The advantage uses a learned value function to approximate the expected return of the current state, and subtracts this from the actual return to get the advantage. This tells us how much better (or worse) the current action was compared to the average action in that state.

The exact math is:

A_t = Q(s_t, a_t) - V(s_t) but Q is approximated by G_t

They also similar to APE-X have different exploration policies per worker (when doing q learning), instead of a deterministic epsilon at the start of training like APE-X they sample an epsilon for each worker from a distribution periodically throughout training. This has similar benefits to APE-X's multiple exploration strategies in that it provides more diverse data for updates.


NOTE: perhaps this should be in VPG?
NOTE: This does not implement the LSTM version which was also described in the A3C Paper or the equivalent Sarsa and DQN versions of the A3C algorithm.
"""

from atomic_rl.initialization import layer_init, set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb
from functools import partial

from atomic_rl.action_selection import sample_distribution
from atomic_rl.optimizer import apply_gradients
from atomic_rl.returns import compute_n_step_returns
from atomic_rl.losses import policy_gradient_loss, huber_loss, mse_loss, entropy_loss
from torch.optim.lr_scheduler import LinearLR
from atomic_rl.metrics import compute_explained_variance
from atomic_rl.buffers.rollout import (
    init_rollout_buffer,
    store_rollout_step_,
    flatten_rollout_buffer,
    record_truncations_,
    get_rollout_next_values,
)
from atomic_rl.utils import (
    standardize_tensor,
    to_tensor,
    to_numpy_action,
)
from tensordict import TensorDict

# Constants
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 10_000
GAMMA = 0.99
ENTROPY_COEFF = 0.001
CRITIC_COEFF = 0.5
MAX_GRAD_NORM = 0.5
N_STEP = 5
STEPS_PER_ENV = N_STEP  # Steps for rollout collection is the same as n_step for the bootstrapping in the A3C paper, though they could be different in practice.
NUM_ENVS = 16
SEED = 42

# Seeding for reproducibility
set_seed(SEED)


# NOTE: we use a fused backbone and separate heads in accordance with the A3C paper
class ActorCritic(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = layer_init(nn.Linear(input_shape[0], 64))
        self.l2 = layer_init(nn.Linear(64, 64))
        self.l3_actor = layer_init(nn.Linear(64, num_actions), std=0.01)
        self.l3_critic = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x_actor = self.l3_actor(x)
        x_critic = self.l3_critic(x)
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


# Initialize vectorized environments using standard Gymnasium Vector Envs
# TODO: switch to either procgen, envpool, isaac gym, brax, or pufferlib to decrease training time (same number iterations but faster)
envs = gym.vector.SyncVectorEnv(
    [make_env("CartPole-v1", SEED + i, i) for i in range(NUM_ENVS)]
)

obs_shape = envs.single_observation_space.shape
num_actions = envs.single_action_space.n
device = torch.device("cpu")

model = ActorCritic(obs_shape, num_actions).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Linearly decay LR from 1.0 * LEARNING_RATE to 0.0 * LEARNING_RATE over MAX_ITERATIONS
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
wandb.init(project="a2c-cartpole", config={"lr": LEARNING_RATE, "gamma": GAMMA})
wandb.define_metric("*", step_metric="global_step")
global_step = 0

# Using full episodes
for iteration in range(MAX_ITERATIONS):
    # NOTE: here we use torch.inference_mode() and do a re-eval pass to compute necessary data, but the re-eval could be integrated into the data collection loop and use python lists similar to VPG. We chose the re-eval pass instead in order to demonstrate the buffer system, and have a parallel to PPO's re-eval pass.
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
            store_rollout_step_(buffer=buffer, step=step, transition=transition)

            # 4. Handle Truncations (Gymnasium auto-resets)
            if "final_observation" in info:
                from atomic_rl.utils import extract_vector_env_final_obs

                env_indices, final_obs = extract_vector_env_final_obs(info)
                # Filter to only record environments that were truncated
                trunc_mask = truncated[env_indices]
                if trunc_mask.any():
                    record_truncations_(
                        buffer,
                        step,
                        torch.as_tensor(
                            env_indices[trunc_mask], dtype=torch.long, device=device
                        ),
                        torch.as_tensor(
                            final_obs[trunc_mask], dtype=torch.float32, device=device
                        ),
                    )
            # Update state for next tick

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

        # Compute last values for the re-evaluation pass
        last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        _, last_values = model(last_obs_tensor)

        # Calculate Loss & Gradients
        next_values = get_rollout_next_values(
            buffer, last_values, get_value_fn=lambda obs: model(obs)[1], device=device
        )

    # --- 3. The Update Loop ---
    # TODO: could we change what return and advantage functions expect so we dont need to squeeze here?
    returns = compute_n_step_returns(
        rewards=buffer.data["rewards"],
        terminated=buffer.data["terminated"],
        truncated=buffer.data["truncated"],
        values=buffer.data["values"].squeeze(-1),
        next_values=next_values.squeeze(-1),
        gamma=GAMMA,
        n=N_STEP,
    )

    # 2. Define Baseline & Calculate Raw Advantage (Explicit Math)
    baseline = buffer.data["values"].detach().squeeze(-1)
    advantages = returns - baseline

    # 3. Optional Scaling
    advantages = standardize_tensor(advantages)

    # Flatten buffer for loss calculations
    flat_data = flatten_rollout_buffer(buffer)
    flat_advantages = advantages.view(-1, 1)
    flat_returns = returns.view(-1, 1)

    # --- Re-evaluation Pass ---
    new_logits, new_values = model(flat_data["observations"])

    # Re-calculate log probabilities for the actions taken
    dist = torch.distributions.Categorical(logits=new_logits)
    # dist has batch shape [B]. flat_data["actions"] is [B, 1]. Squeeze for log_prob, unsqueeze output.
    new_log_probs = dist.log_prob(flat_data["actions"].squeeze(-1)).unsqueeze(-1)

    pg_loss, info_dict = policy_gradient_loss(
        advantages=flat_advantages,
        log_probs=new_log_probs,
    )

    ent_loss, _ = entropy_loss(dist)
    ent_loss = ent_loss.mean()

    pg_loss = pg_loss.mean()

    # NOTE: i use my mse_loss from losses.py, but we don't need priorities here since it's not off policy.
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
    # TODO: do this here or start of collection phase?
    buffer.truncation_records.clear()
    # buffer.data.detach_()

    if iteration % 100 == 0:
        # Calculate explained variance
        explained_var = compute_explained_variance(
            returns.detach().cpu().numpy(), buffer.data["values"].detach().cpu().numpy()
        )

        # W&B handles scalars and histograms of tensors (like priorities) automatically.
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
                "advantages/std": advantages.std().item(),
                "global_step": global_step,
            }
        )
        wandb.log(log_dict)

wandb.finish()
