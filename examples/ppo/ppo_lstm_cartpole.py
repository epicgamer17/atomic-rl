from atomic_rl.initialization import layer_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple, List, Optional, Callable
import numpy as np
import random
import wandb
from tensordict import TensorDict

from atomic_rl.action_selection import sample_distribution
from atomic_rl.optimizer import apply_gradients
from atomic_rl.returns import compute_gae
from atomic_rl.losses import (
    clipped_surrogate_loss,
    entropy_loss,
    probability_ratio,
    clipped_mse_loss,
)
from torch.optim.lr_scheduler import LinearLR
from atomic_rl.visualization import compute_explained_variance
from atomic_rl.network import unroll_rnn
from atomic_rl.rollout_buffer import (
    init_rollout_buffer,
    store_rollout_step_,
    flatten_rollout_buffer,
    record_truncations_,
    get_rollout_next_values,
    yield_shuffled_minibatches,
    yield_sequential_minibatches,
)
from atomic_rl.utils import (
    standardize_tensor,
    to_tensor,
    to_numpy_action,
    extract_vector_env_final_obs,
)
from envs.wrappers import FlickeringObservation, NormalizeObservation, VecNormalize

# TODO: is this working PPO + LSTM fails to learn MDP cartpole (it performs worse than vanilla PPO)

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
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(SEED)


class ActorCritic(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.actor = nn.Sequential(
            # TODO: layer_init does not take in an RNG key but should for reproducibility
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

    def forward(self, x: torch.Tensor):
        return self.actor(x), self.critic(x)


class ActorCriticLSTM(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        # Separate Actor Path
        self.actor_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_lstm = nn.LSTM(64, 64)
        for name, param in self.actor_lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor_head = layer_init(nn.Linear(64, num_actions), std=0.01)

        # Separate Critic Path
        self.critic_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.critic_lstm = nn.LSTM(64, 64)
        for name, param in self.critic_lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.critic_head = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(
        self,
        x: torch.Tensor,
        lstm_state: Tuple[torch.Tensor, torch.Tensor],
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # lstm_state is [actor_h, actor_c, critic_h, critic_c]
        actor_h, actor_c, critic_h, critic_c = lstm_state

        # B = batch size, T = sequence length
        B = actor_h.shape[1]

        # Safety check: if x.shape[0] isn't a multiple of B, we assume x is a batch of observations
        # for a different number of environments than what's in the current lstm_state.
        # TODO: what is this seems hacky. should fix.
        if x.shape[0] % B != 0:
            B = x.shape[0]
            actor_h = torch.zeros(
                self.actor_lstm.num_layers,
                B,
                self.actor_lstm.hidden_size,
                device=x.device,
            )
            actor_c = torch.zeros(
                self.actor_lstm.num_layers,
                B,
                self.actor_lstm.hidden_size,
                device=x.device,
            )
            critic_h = torch.zeros(
                self.critic_lstm.num_layers,
                B,
                self.critic_lstm.hidden_size,
                device=x.device,
            )
            critic_c = torch.zeros(
                self.critic_lstm.num_layers,
                B,
                self.critic_lstm.hidden_size,
                device=x.device,
            )
            dones = torch.zeros(B, device=x.device)

        T = x.shape[0] // B

        # Feature Extraction
        actor_hidden = self.actor_feature_extractor(x)
        critic_hidden = self.critic_feature_extractor(x)

        # Prepare for LSTM: [T*B, F] -> [B, T, F]
        actor_hidden = actor_hidden.reshape(T, B, -1).transpose(0, 1)
        critic_hidden = critic_hidden.reshape(T, B, -1).transpose(0, 1)
        mb_dones = dones.reshape(T, B).transpose(0, 1)

        # Unroll LSTMs
        actor_hidden_seq, (actor_h, actor_c) = unroll_rnn(
            self.actor_lstm, actor_hidden, (actor_h, actor_c), mb_dones
        )
        critic_hidden_seq, (critic_h, critic_c) = unroll_rnn(
            self.critic_lstm, critic_hidden, (critic_h, critic_c), mb_dones
        )

        # Re-flatten: [B, T, F] -> [T*B, F]
        actor_hidden_seq = actor_hidden_seq.transpose(0, 1).reshape(
            -1, self.actor_lstm.hidden_size
        )
        critic_hidden_seq = critic_hidden_seq.transpose(0, 1).reshape(
            -1, self.critic_lstm.hidden_size
        )

        logits = self.actor_head(actor_hidden_seq)
        value = self.critic_head(critic_hidden_seq)

        return logits, value, (actor_h, actor_c, critic_h, critic_c)


def make_env(env_id, seed, idx, flickering_prob: float = 0.0):
    def thunk():
        env = gym.make(env_id)
        if flickering_prob > 0:
            env = FlickeringObservation(env, prob=flickering_prob)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# TODO: maybe remove this train_ppo function since its not the convention in our code, or update the code to make this the convention
def train_ppo(use_lstm: bool = False):
    device = torch.device("cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", SEED + i, i) for i in range(NUM_ENVS)]
    )
    # envs = VecNormalize(
    #     envs,
    #     norm_obs=True,
    #     norm_reward=True,
    #     clip_obs=10.0,
    #     clip_reward=10.0,
    #     gamma=GAMMA,
    # )
    obs_shape = envs.single_observation_space.shape
    num_actions = envs.single_action_space.n

    if use_lstm:
        model = ActorCriticLSTM(obs_shape, num_actions).to(device)
    else:
        model = ActorCritic(obs_shape, num_actions).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)
    scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERATIONS
    )

    obs, info = envs.reset(seed=SEED)

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
    # TODO: clean this up in all lstm buffers, we have terminated truncated and dones.
    if use_lstm:
        shapes["dones"] = ()

    buffer = init_rollout_buffer(
        steps_per_env=STEPS_PER_ENV,
        num_envs=NUM_ENVS,
        shapes=shapes,
        device=device,
    )

    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)

    wandb.init(
        project="ppo-lstm-cartpole",
        name=f"ppo_{'lstm' if use_lstm else 'mlp'}",
        config={
            "lr": LEARNING_RATE,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "update_epochs": UPDATE_EPOCHS,
            "minibatch_size": MINIBATCH_SIZE,
            "clip_coef": CLIP_COEF,
            "num_envs": NUM_ENVS,
            "steps_per_env": STEPS_PER_ENV,
            "use_lstm": use_lstm,
        },
        reinit=True,
    )
    wandb.define_metric("*", step_metric="global_step")
    global_step = 0

    if use_lstm:
        # Initialize persistent LSTM states (Actor and Critic separate)
        next_lstm_state = (
            torch.zeros(
                model.actor_lstm.num_layers,
                NUM_ENVS,
                model.actor_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.actor_lstm.num_layers,
                NUM_ENVS,
                model.actor_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.critic_lstm.num_layers,
                NUM_ENVS,
                model.critic_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.critic_lstm.num_layers,
                NUM_ENVS,
                model.critic_lstm.hidden_size,
                device=device,
            ),
        )
        # A dummy "done" flag for step 0
        next_done = torch.zeros(NUM_ENVS, device=device)

    # Training Loop
    for iteration in range(MAX_ITERATIONS):
        # Cache the initial LSTM state for the optimization phase
        if use_lstm:
            initial_lstm_state = tuple(s.clone() for s in next_lstm_state)

        with torch.inference_mode():
            for step in range(STEPS_PER_ENV):
                obs_tensor = to_tensor(obs, device=device)
                if use_lstm:
                    logits, value, next_lstm_state = model(
                        obs_tensor, next_lstm_state, next_done
                    )
                else:
                    logits, value = model(obs_tensor)

                dist = torch.distributions.Categorical(logits=logits)
                action, info_dict = sample_distribution(dist, explore=True)
                action_np = to_numpy_action(action)

                next_obs, reward, terminated, truncated, info = envs.step(action_np)
                global_step += NUM_ENVS

                transition_data = {
                    "observations": obs_tensor,
                    "actions": action,
                    "logprobs": info_dict["log_prob"].detach(),
                    "rewards": to_tensor(reward, device=device),
                    "terminated": to_tensor(terminated, device=device),
                    "truncated": to_tensor(truncated, device=device),
                    "values": value.detach(),
                    "logits": logits.detach(),
                }
                if use_lstm:
                    transition_data["dones"] = next_done

                # TODO: URGENT. Creating a new tensor every step in the hotloop. should do in a more efficient way, maybe some thing like the rollout buffer in PPO (possible reuse?) ie store in a rollout buffer before sending to main replay buffer. Idea being its pre allocated basically. Must consider the N-Step case.
                transition = TensorDict(transition_data, batch_size=[NUM_ENVS])
                store_rollout_step_(buffer=buffer, step=step, transition=transition)

                if "final_observation" in info:
                    env_indices, final_obs = extract_vector_env_final_obs(info)
                    trunc_mask = truncated[env_indices]
                    if trunc_mask.any():
                        # TODO: We should correctly handle LSTM hidden states on bootstrapping truncated states.
                        # Currently we only store the final observation, but the hidden state should also be preserved
                        # or re-computed to get an accurate value estimate for the truncated state.
                        record_truncations_(
                            buffer,
                            step,
                            torch.as_tensor(
                                env_indices[trunc_mask], dtype=torch.long, device=device
                            ),
                            torch.as_tensor(
                                final_obs[trunc_mask],
                                dtype=torch.float32,
                                device=device,
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
                if use_lstm:
                    next_done = torch.as_tensor(
                        terminated | truncated, dtype=torch.float32, device=device
                    )

            last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if use_lstm:
                _, last_values, _ = model(last_obs_tensor, next_lstm_state, next_done)

                # TODO: this is not correct, we should somehow get the value for the truncated states ideally using the lstm state for that truncated state and not an arbitrary one. DO THIS IN ALL LSTM EXAMPLES WITH BOOTSTRAPPING.
                def get_value_fn(o):
                    N = o.shape[0]
                    # Create zero states matching the number of truncated obs
                    zero_states = (
                        torch.zeros(
                            model.actor_lstm.num_layers,
                            N,
                            model.actor_lstm.hidden_size,
                            device=device,
                        ),
                        torch.zeros(
                            model.actor_lstm.num_layers,
                            N,
                            model.actor_lstm.hidden_size,
                            device=device,
                        ),
                        torch.zeros(
                            model.critic_lstm.num_layers,
                            N,
                            model.critic_lstm.hidden_size,
                            device=device,
                        ),
                        torch.zeros(
                            model.critic_lstm.num_layers,
                            N,
                            model.critic_lstm.hidden_size,
                            device=device,
                        ),
                    )
                    zero_dones = torch.zeros(N, device=device)
                    return model(o, zero_states, zero_dones)[1]

            else:
                _, last_values = model(last_obs_tensor)
                get_value_fn = lambda o: model(o)[1]

            next_values = get_rollout_next_values(
                buffer, last_values, get_value_fn=get_value_fn, device=device
            )

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

        buffer.data["advantages"] = advantages.unsqueeze(-1)
        buffer.data["returns"] = returns

        epoch_losses = []
        clip_fractions = []
        approx_kls = []

        for epoch in range(UPDATE_EPOCHS):
            epoch_kls = []
            if use_lstm:
                minibatch_generator = yield_sequential_minibatches(
                    buffer.data,
                    num_envs=NUM_ENVS,
                    num_minibatches=4,
                    initial_lstm_states=initial_lstm_state,
                    generator=rng_key,
                )
            else:
                flat_data = flatten_rollout_buffer(buffer)
                flat_data["advantages"] = advantages.view(-1, 1)
                flat_data["returns"] = returns.view(-1, 1)
                minibatch_generator = (
                    (mb, None)
                    for mb in yield_shuffled_minibatches(
                        flat_data, MINIBATCH_SIZE, generator=rng_key
                    )
                )

            for mb, mb_initial_lstm_state in minibatch_generator:
                if use_lstm:
                    new_logits, new_values, _ = model(
                        mb["observations"], mb_initial_lstm_state, mb["dones"]
                    )
                else:
                    new_logits, new_values = model(mb["observations"])

                dist = torch.distributions.Categorical(logits=new_logits)
                # dist has batch shape [B]. mb["actions"] is [B, 1]. Squeeze for log_prob, unsqueeze output.
                new_log_probs = dist.log_prob(mb["actions"].squeeze(-1)).unsqueeze(-1)

                ratio = probability_ratio(
                    old_log_probs=mb["logprobs"], new_log_probs=new_log_probs
                )
                mb_advantages = standardize_tensor(mb["advantages"])

                pg_loss, pg_info = clipped_surrogate_loss(
                    ratio=ratio, advantages=mb_advantages, clip_coef=CLIP_COEF
                )
                pg_loss = pg_loss.mean()

                critic_loss, _ = clipped_mse_loss(
                    predictions=new_values,
                    old_predictions=mb["values"],
                    targets=mb["returns"],
                    clip_coef=CLIP_COEF,
                )
                critic_loss = critic_loss.mean()

                ent_loss, _ = entropy_loss(dist)
                ent_loss = ent_loss.mean()

                loss = pg_loss + CRITIC_COEFF * critic_loss + ENTROPY_COEFF * ent_loss
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

        scheduler.step()
        buffer.truncation_records.clear()

        if iteration % 10 == 0:
            flat_returns = returns.flatten()
            flat_values = buffer.data["values"].flatten()
            explained_var = compute_explained_variance(
                flat_returns.detach().cpu().numpy(),
                flat_values.detach().cpu().numpy(),
            )

            log_dict = {
                "learning_rate": scheduler.get_last_lr()[0],
                "loss/total": np.mean(epoch_losses),
                "loss/critic": critic_loss.item(),
                "loss/policy": pg_loss.item(),
                "loss/entropy": ent_loss.item(),
                "value/mean": buffer.data["values"].mean().item(),
                "value/return_mean": returns.mean().item(),
                "value/explained_variance": explained_var,
                "advantages/mean": advantages.mean().item(),
                "advantages/std": advantages.std().item(),
                "ppo/clip_fraction": np.mean(clip_fractions),
                "ppo/approx_kl": np.mean(approx_kls),
                "global_step": global_step,
            }
            wandb.log(log_dict)

            if iteration % 100 == 0:
                print(
                    f"Iteration {iteration}, Global Step {global_step}, Avg Loss {np.mean(epoch_losses)}"
                )

    wandb.finish()
    envs.close()
    return model


def evaluate_model(
    model, use_lstm: bool, flickering_prob: float = 0.5, num_episodes: int = 10
):
    device = torch.device("cpu")
    # We use a single env for evaluation to keep it simple
    env = make_env("CartPole-v1", SEED + 100, 0, flickering_prob=flickering_prob)()

    total_returns = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0

        if use_lstm:
            lstm_state = (
                torch.zeros(
                    model.actor_lstm.num_layers,
                    1,
                    model.actor_lstm.hidden_size,
                    device=device,
                ),
                torch.zeros(
                    model.actor_lstm.num_layers,
                    1,
                    model.actor_lstm.hidden_size,
                    device=device,
                ),
                torch.zeros(
                    model.critic_lstm.num_layers,
                    1,
                    model.critic_lstm.hidden_size,
                    device=device,
                ),
                torch.zeros(
                    model.critic_lstm.num_layers,
                    1,
                    model.critic_lstm.hidden_size,
                    device=device,
                ),
            )
            dones = torch.zeros(1, device=device)

        while not done:
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.inference_mode():
                if use_lstm:
                    logits, _, lstm_state = model(obs_tensor, lstm_state, dones)
                else:
                    logits, _ = model(obs_tensor)

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

            if use_lstm:
                dones = torch.as_tensor([done], dtype=torch.float32, device=device)

        total_returns.append(ep_return)

    env.close()
    return np.mean(total_returns)


if __name__ == "__main__":
    print("\nTraining PPO + LSTM...")
    lstm_model = train_ppo(use_lstm=True)

    print("Training Standard PPO...")
    mlp_model = train_ppo(use_lstm=False)

    # Evaluation
    probs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    print("\nEvaluating models on flickering observations...")
    print(f"{'Prob':<10} | {'MLP Return':<12} | {'LSTM Return':<12}")
    print("-" * 40)

    results = []
    for p in probs:
        mlp_ret = evaluate_model(mlp_model, use_lstm=False, flickering_prob=p)
        lstm_ret = evaluate_model(lstm_model, use_lstm=True, flickering_prob=p)
        print(f"{p:<10.1f} | {mlp_ret:<12.2f} | {lstm_ret:<12.2f}")
        results.append((p, mlp_ret, lstm_ret))

    # Log comparison to wandb
    wandb.init(project="ppo-lstm-cartpole", name="comparison")
    for p, mlp_ret, lstm_ret in results:
        wandb.log(
            {
                "eval/flicker_prob": p,
                "eval/mlp_return": mlp_ret,
                "eval/lstm_return": lstm_ret,
            }
        )
    wandb.finish()
