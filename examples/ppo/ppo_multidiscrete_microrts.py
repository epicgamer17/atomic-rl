# Fully Generated
# TODO: compare with 37 implementation details of PPO results
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb
from einops import rearrange
from tensordict import TensorDict
from functools import partial

from functional.action_selection import (
    multidiscrete_sampling_selector,
    apply_action_mask,
    compute_masked_entropy,
)
from functional.optimizer import apply_gradients
from functional.returns import compute_gae, compute_td_lambda_returns
from functional.losses import (
    clipped_surrogate_loss,
    entropy_loss,
    probability_ratio,
    clipped_value_loss,
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

# Constants
LEARNING_RATE = 2.5e-4
MAX_ITERATIONS = 1953
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.01
CRITIC_COEFF = 0.5
MAX_GRAD_NORM = 0.5
STEPS_PER_ENV = 128
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
CLIP_COEF = 0.1
TARGET_KL = None
NUM_ENVS = 8
SEED = 42

# Seeding
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class Transpose(nn.Module):
    """Permutes tensor dimensions. Used to convert (B, H, W, C) to (B, C, H, W)."""

    def __init__(self, permutation: Tuple[int, ...]):
        super().__init__()
        self.permutation = permutation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.permutation)


class ActorCriticMicroRTS(nn.Module):
    def __init__(self, nvec: Tuple[int, ...]):
        super().__init__()
        self.nvec = nvec

        # Feature extractor with explicit Transpose for MicroRTS
        # Theory: MicroRTS observations are usually provided as [H, W, C].
        # CNNs in PyTorch expect [C, H, W], so we transpose first.
        self.network = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 3 * 3, 128)),
            nn.ReLU(),
        )

        # Actor outputs a flat tensor of size sum(nvec)
        # Each segment of the output will represent logits for one discrete action component.
        self.actor = layer_init(nn.Linear(128, sum(self.nvec)), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network(x)
        return self.actor(hidden), self.critic(hidden)


def make_env(env_id: str, seed: int):
    def thunk():
        # NOTE: MicroRTS requires specific wrappers and gym-microrts installation.
        # We use a mock MultiDiscrete environment here if the library is not found.
        try:
            import gym_microrts  # type: ignore

            env = gym.make(env_id)
        except ImportError:
            # Mock environment with MicroRTS-like dimensions
            # Obs: [10, 10, 27], Actions: MultiDiscrete([2, 3, 4])
            env = gym.wrappers.TimeLimit(
                gym.envs.registration.make("CartPole-v1"), max_episode_steps=200
            )
            # Override spaces for mock
            env.observation_space = gym.spaces.Box(0, 1, (10, 10, 27), dtype=np.float32)
            env.action_space = gym.spaces.MultiDiscrete([2, 3, 4])

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# TODO: switch to either procgen, envpool, isaac gym, brax, or pufferlib to decrease training time (same number iterations but faster)
# Initialize vectorized environments
ENV_ID = "MicroRTS-v0"  # Example ID
envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, SEED + i) for i in range(NUM_ENVS)])

obs_shape = envs.single_observation_space.shape
nvec = tuple(envs.single_action_space.nvec)
device = torch.device("cpu")  # Use CUDA if available

model = ActorCriticMicroRTS(nvec).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)
scheduler = LinearLR(
    optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERATIONS
)

obs, info = envs.reset(seed=SEED)

# Pre-allocate rollout buffers
shapes = {
    "observations": obs_shape,
    "actions": (len(nvec),),  # One action per discrete sub-space
    "logprobs": (),
    "rewards": (),
    "terminated": (),
    "truncated": (),
    "values": (),
    "logits": (sum(nvec),),  # Logits is the flat sum of all sub-spaces
    "action_masks": (sum(nvec),),  # Store the flat mask
}
buffer = init_rollout_buffer(
    steps_per_env=STEPS_PER_ENV,
    num_envs=NUM_ENVS,
    shapes=shapes,
    device=device,
)

rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

# Initialize W&B
wandb.init(
    project="ppo-multidiscrete-microrts",
    config={
        "lr": LEARNING_RATE,
        "gamma": GAMMA,
        "update_epochs": UPDATE_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "num_envs": NUM_ENVS,
        "nvec": nvec,
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
            logits, value = model(obs_tensor)

            # Get the action mask from MicroRTS environments
            try:
                action_mask = torch.as_tensor(
                    envs.call("get_action_mask"), dtype=torch.float32, device=device
                )
            except Exception:
                # Mock mask for demonstration if get_action_mask fails
                action_mask = torch.ones(
                    (NUM_ENVS, sum(nvec)), dtype=torch.float32, device=device
                )

            # Apply the mask to logits before sampling
            masked_logits = apply_action_mask(logits, action_mask)

            # Use the MultiDiscrete functional selector with masked logits
            action, info_dict = multidiscrete_sampling_selector(
                masked_logits, nvec, temperature=1.0
            )
            # Step env expects numpy array (batch_size, len(nvec))
            action_np = action.cpu().numpy().astype(np.int32)

            next_obs, reward, terminated, truncated, info = envs.step(action_np)
            global_step += NUM_ENVS

            # Store in rollout buffer
            transition = TensorDict(
                {
                    "observations": obs_tensor,
                    "actions": action,  # [Batch, len(nvec)]
                    "logprobs": info_dict["log_prob"].detach(),  # [Batch]
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
                    "logits": logits.detach(),
                    "action_masks": action_mask.detach(),
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

        # Compute last values for GAE
        last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        _, last_values = model(last_obs_tensor)
        last_values = last_values.squeeze(-1)

        next_values = get_rollout_next_values(
            buffer, last_values, get_value_fn=lambda obs: model(obs)[1], device=device
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
    for epoch in range(UPDATE_EPOCHS):
        epoch_kls = []
        for mb in yield_shuffled_minibatches(
            flat_data, MINIBATCH_SIZE, generator=rng_key
        ):
            new_logits, new_values = model(mb["observations"])
            new_values = new_values.squeeze(-1)

            # 1. Apply the mask functionally to the raw logits BEFORE splitting
            masked_logits = apply_action_mask(new_logits, mb["action_masks"])

            # 2. Split logits and masks
            split_logits = torch.split(masked_logits, list(nvec), dim=-1)
            split_masks = torch.split(mb["action_masks"], list(nvec), dim=-1)

            new_log_probs_list = []
            entropy_list = []

            for i, (component_logits, component_mask) in enumerate(
                zip(split_logits, split_masks)
            ):
                dist = torch.distributions.Categorical(logits=component_logits)

                # Log Prob
                new_log_probs_list.append(dist.log_prob(mb["actions"][:, i]))

                # Explicit Masked Entropy calculation
                component_entropy = compute_masked_entropy(
                    logits=component_logits, probs=dist.probs, mask=component_mask
                )
                entropy_list.append(component_entropy)

            # 3. Sum components (Independent action components theory)
            new_log_probs = torch.stack(new_log_probs_list, dim=-1).sum(dim=-1)
            total_entropy = torch.stack(entropy_list, dim=-1).sum(dim=-1)

            # Compute KL divergence
            with torch.no_grad():
                log_ratio = new_log_probs - mb["logprobs"]
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
            epoch_kls.append(approx_kl)

            # 2. Policy Loss (Clipped Surrogate)
            ratio = probability_ratio(
                old_log_probs=mb["logprobs"], new_log_probs=new_log_probs
            )
            mb_advantages = standardize_tensor(mb["advantages"])
            pg_loss, _ = clipped_surrogate_loss(ratio, mb_advantages, CLIP_COEF)

            # 3. Value Loss
            critic_loss, _ = clipped_value_loss(
                new_values, mb["values"], mb["returns"], CLIP_COEF
            )

            # Total Loss (using the explicitly summed entropy)
            loss = (
                pg_loss.mean()
                + CRITIC_COEFF * critic_loss.mean()
                - ENTROPY_COEFF * total_entropy.mean()
            )

            # Apply gradients via functional optimizer
            optimizer = apply_gradients(
                optimizer, loss, model=model, clip_grad_norm=MAX_GRAD_NORM
            )
            epoch_losses.append(loss.item())

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
        wandb.log(
            {
                "learning_rate": scheduler.get_last_lr()[0],
                "loss/total": np.mean(epoch_losses),
                "value/explained_variance": explained_var,
                "global_step": global_step,
            }
        )

wandb.finish()
