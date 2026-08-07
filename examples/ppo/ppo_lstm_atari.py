# Fully Generated
# TODO: compare with 37 implementation details of PPO results
# TODO: attempt a cleanup if possible
# TODO: notes on PPO + LSTM
from functional.initialization import layer_init, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb

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
from functional.network import unroll_rnn
from functional.rollout_buffer import (
    init_rollout_buffer,
    store_rollout_step,
    store_rollout_step_,
    record_truncations_,
    get_rollout_next_values,
    yield_shuffled_minibatches,
    yield_sequential_minibatches,
)
from functional.utils import (
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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class ActorCriticLSTM(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        # Nature CNN feature extractor from networks layer
        self.network = AtariCNN(in_channels=4, out_features=512, scale_inputs=False)

        # 2. LSTM Layer
        self.lstm = nn.LSTM(512, 128)

        # LSTM Initialization (Implementation Detail)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # 3. Output Heads
        self.actor = layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def forward(
        self,
        x: torch.Tensor,
        lstm_state: Tuple[torch.Tensor, torch.Tensor],
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x shape: [sequence_length * batch_size, channels, h, w]
        lstm_state shape: ( [1, batch_size, 128], [1, batch_size, 128] )
        dones shape: [sequence_length * batch_size]
        """
        # Feature extraction
        hidden = self.network(x / 255.0)

        # Recover Sequence and Batch dimensions
        batch_size = lstm_state[0].shape[1]
        T = hidden.shape[0] // batch_size

        # [seq_len * batch, features] -> [batch, seq_len, features]
        hidden = hidden.reshape(T, batch_size, -1).transpose(0, 1)
        dones = dones.reshape(T, batch_size).transpose(0, 1)

        # Unroll LSTM with state resets on 'done' steps
        new_hidden_sequence, lstm_state = unroll_rnn(
            self.lstm, hidden, lstm_state, dones
        )

        # Re-flatten back to [seq_len * batch, features]
        new_hidden_sequence = new_hidden_sequence.transpose(0, 1).reshape(
            -1, self.lstm.hidden_size
        )

        return (
            self.actor(new_hidden_sequence),
            self.critic(new_hidden_sequence),
            lstm_state,
        )


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

model = ActorCriticLSTM(num_actions).to(device)

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
    "dones": (),
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

# Initialize persistent LSTM states
next_lstm_state = (
    torch.zeros(model.lstm.num_layers, NUM_ENVS, model.lstm.hidden_size, device=device),
    torch.zeros(model.lstm.num_layers, NUM_ENVS, model.lstm.hidden_size, device=device),
)
# A dummy "done" flag for step 0
next_done = torch.zeros(NUM_ENVS, device=device)

# Training Loop
for iteration in range(MAX_ITERATIONS):
    # Cache the initial LSTM state for the optimization phase
    initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

    # 1. Data Collection Phase
    with torch.inference_mode():
        for step in range(STEPS_PER_ENV):
            # Atari observations are [C, H, W] after wrappers
            obs_tensor = to_tensor(obs, device=device)
            logits, value, next_lstm_state = model(
                obs_tensor, next_lstm_state, next_done
            )

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
                    "dones": next_done,
                },
                batch_size=[NUM_ENVS],
            )
            store_rollout_step_(buffer=buffer, step=step, transition=transition)

            # 4. Handle Truncations (Gymnasium auto-resets)
            if "final_observation" in info:
                from functional.utils import extract_vector_env_final_obs

                env_indices, final_obs = extract_vector_env_final_obs(info)
                # Filter to only record environments that were truncated
                trunc_mask = truncated[env_indices]
                if trunc_mask.any():
                    # TODO: We should correctly handle LSTM hidden states on bootstrapping truncated states.
                    # Currently we only store the final observation, but the hidden state should also be preserved
                    # or re-computed to get an accurate value estimate for the truncated state.
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
            next_done = torch.as_tensor(
                terminated | truncated, dtype=torch.float32, device=device
            )

        # Compute last values for GAE
        last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        _, last_values, _ = model(last_obs_tensor, next_lstm_state, next_done)

        next_values = get_rollout_next_values(
            buffer,
            last_values,
            # TODO: this is not correct, we should somehow get the value for the truncated states ideally using the lstm state for that truncated state and not an arbitrary one. DO THIS IN ALL LSTM EXAMPLES WITH BOOTSTRAPPING.
            get_value_fn=lambda obs: model(
                obs,
                (
                    torch.zeros(
                        model.lstm.num_layers,
                        obs.shape[0],
                        model.lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.lstm.num_layers,
                        obs.shape[0],
                        model.lstm.hidden_size,
                        device=device,
                    ),
                ),
                torch.zeros(obs.shape[0], device=device),
            )[1],
            device=device,
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

    # Add advantages and returns to the buffer data for sequential sampling
    buffer.data["advantages"] = advantages.unsqueeze(-1)
    buffer.data["returns"] = returns

    # 3. Optimization Phase
    epoch_losses = []
    clip_fractions = []
    approx_kls = []

    for epoch in range(UPDATE_EPOCHS):
        epoch_kls = []
        for mb, mb_initial_lstm_state in yield_sequential_minibatches(
            buffer.data,
            num_envs=NUM_ENVS,
            num_minibatches=4,
            initial_lstm_states=initial_lstm_state,
            generator=rng_key,
        ):
            # Re-evaluate model on minibatch
            new_logits, new_values, _ = model(
                mb["observations"], mb_initial_lstm_state, mb["dones"]
            )

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
            buffer.data["returns"].flatten().detach().cpu().numpy(),
            buffer.data["values"].flatten().detach().cpu().numpy(),
        )

        log_dict = {
            "learning_rate": scheduler.get_last_lr()[0],
            "loss/total": np.mean(epoch_losses),
            "loss/critic": critic_loss.item(),
            "loss/policy": pg_loss.item(),
            "loss/entropy": ent_loss.item(),
            "value/mean": buffer.data["values"].mean().item(),
            "value/return_mean": buffer.data["returns"].mean().item(),
            "value/explained_variance": explained_var,
            "advantages/mean": advantages.mean().item(),
            "advantages/std": advantages.std().item(),
            "ppo/clip_fraction": np.mean(clip_fractions),
            "ppo/approx_kl": np.mean(approx_kls),
            "global_step": global_step,
        }
        wandb.log(log_dict)

wandb.finish()
