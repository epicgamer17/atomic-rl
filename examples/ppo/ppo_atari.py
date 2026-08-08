# Fully Generated
# TODO: compare with 37 implementation details of PPO results
# TODO: attempt a cleanup if possible
from atomic_rl.initialization import layer_init, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb

from atomic_rl.action_selection import sample_distribution
from atomic_rl.optimizer import apply_gradients
from atomic_rl.returns import compute_gae
from atomic_rl.losses import (
    clipped_surrogate_loss,
    entropy_loss,
    probability_ratio,
    clipped_mse_loss,
    mse_loss,
)
from torch.optim.lr_scheduler import LinearLR
from atomic_rl.metrics import compute_explained_variance
from atomic_rl.buffers.rollout import (
    init_rollout_buffer,
    store_rollout_step_,
    record_truncations_,
    get_rollout_next_values,
    yield_shuffled_minibatches,
)
from atomic_rl.utils import (
    ema_update,
    standardize_tensor,
    to_tensor,
    to_numpy_action,
)
from envs.wrappers import FireResetEnv
from networks import AtariCNN
from tensordict import TensorDict

# Constants
LEARNING_RATE = 2.5e-4
MAX_ITERATIONS = 9765  # Adjust as needed for Pong
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.01
CRITIC_COEFF = 0.5
MAX_GRAD_NORM = 0.5
STEPS_PER_ENV = 128
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
CLIP_COEF = 0.1
TARGET_KL = None  # Usually not used for Atari
NUM_ENVS = 8
SEED = 42

# Seeding
set_seed(SEED)


class ActorCritic(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        # Nature CNN feature extractor from networks layer
        self.network = AtariCNN(in_channels=4, out_features=512, scale_inputs=True)
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        hidden = self.network(x)
        return self.actor(hidden), self.critic(hidden)


def make_env(env_id, seed, idx):
    """
    Standard Atari environment preprocessing.
    - Pong is the default game.
    - Frameskip (4): repeats action on skipped frames, sums rewards (Mnih et al., 2015).
    - MaxPool: handles sprite flickering by taking max over last 2 frames.
    - NoopReset (1-30): injects stochasticity into initial states (Machado et al., 2018).
    - EpisodicLife: marks end-of-life as end-of-episode (Mnih et al., 2015).
    - Grayscale + Resize (84x84): extracts luminance and reduces dimensionality.
    - FrameStack (4): allows agent to infer velocity and direction.
    """

    def thunk():
        env = gym.make(env_id, frameskip=1)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=True,
            grayscale_obs=True,
        )

        # FIRE on reset (necessary for Pong)
        env = FireResetEnv(env)

        # Frame Stack: 4 frames
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# TODO: switch to either procgen, envpool, isaac gym, brax, or pufferlib to decrease training time (same number iterations but faster)
# Initialize vectorized environments using standard Gymnasium Vector Envs
envs = gym.vector.SyncVectorEnv(
    [make_env("ALE/Pong-v5", SEED + i, i) for i in range(NUM_ENVS)]
)

obs_shape = envs.single_observation_space.shape
num_actions = envs.single_action_space.n
device = torch.device("cpu")  # Use CUDA if available

model = ActorCritic(num_actions).to(device)

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
    "logits": (num_actions,),
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
    project="ppo-atari",
    config={
        "lr": LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "update_epochs": UPDATE_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "clip_coef": CLIP_COEF,
        "num_envs": NUM_ENVS,
        "steps_per_env": STEPS_PER_ENV,
        "env": "ALE/Pong-v5",
    },
)
wandb.define_metric("*", step_metric="global_step")
global_step = 0

# Training Loop
for iteration in range(MAX_ITERATIONS):
    # 1. Data Collection Phase
    with torch.inference_mode():
        for step in range(STEPS_PER_ENV):
            # Atari observations are [C, H, W] after wrappers
            obs_tensor = to_tensor(obs, device=device)
            logits, value = model(obs_tensor)

            dist = torch.distributions.Categorical(logits=logits)
            action, info_dict = sample_distribution(dist, explore=True)
            action_np = to_numpy_action(action)

            # Step Env
            next_obs, reward, terminated, truncated, info = envs.step(action_np)
            global_step += NUM_ENVS

            # Reward Clipping: np.sign(reward)
            # Source: Mnih et al. (2015) - Bins reward to {+1, 0, -1}
            clipped_reward = np.sign(reward).astype(np.float32)

            # Store in rollout buffer
            transition = TensorDict(
                {
                    "observations": obs_tensor,
                    "actions": action,
                    "logprobs": info_dict["log_prob"].detach(),
                    "rewards": to_tensor(clipped_reward, device=device),
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
                        buffer=buffer,
                        step=step,
                        truncated_envs=to_tensor(env_indices[trunc_mask], device),
                        final_observations=to_tensor(final_obs[trunc_mask], device),
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
        _, last_values = model(last_obs_tensor)

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

    # Flatten buffer for training
    flat_data = flatten_rollout_buffer(buffer)
    flat_advantages = advantages.view(-1, 1)
    flat_returns = returns.view(-1, 1)

    # Add advantages and returns to flat_data for easier minibatch sampling
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
            new_logits, new_values = model(mb["observations"])

            # Categorical distribution for Atari
            dist = torch.distributions.Categorical(logits=new_logits)
            # dist has batch shape [B]. mb["actions"] is [B, 1]. Squeeze for log_prob, unsqueeze output.
            new_log_probs = dist.log_prob(mb["actions"].squeeze(-1)).unsqueeze(-1)

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

        # Early stopping based on KL (if TARGET_KL is set)
        if TARGET_KL is not None and np.mean(epoch_kls) > 1.5 * TARGET_KL:
            break

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
