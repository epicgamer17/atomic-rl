"""
=============================================================================
MuZero CartPole Implementation Notes & Technical Question Answers
=============================================================================

Required Capabilities (Schrittwieser et al., 2020):
1. Precision planning tasks
2. Visually complex games
3. Single agent domains
4. Non-zero rewards at intermediate steps

Features and Details:
1. Model receives input and transforms it to a hidden state
2. Dynamics iteratively updates hidden state
3. Trained end-to-end
4. Action is sampled from search policy (uses temperature)
5. Targets are n-step returns for MDPs (discounted n-step for value)
6. Dynamics function is deterministic
7. No to-play prediction
8. All parameters trained jointly
9. Policy target is the search policy
10. Value target is the discounted bootstrapped search value
11. Reward loss uses the observed reward (no reward loss on root)
12. Additional L2 loss
13. Cross entropy loss for value and reward on continuous scalar domains (Atari/MDPs)
14. Cross entropy always for policy loss
15. Reanalyze model (reanalyze old trajectories by rerunning MCTS on them using latest network parameters)
16. Metric of "thinking time" for search
17. Evaluation of MCTS with different number of simulations after training
18. MuZero search uses dynamics model instead of perfect simulator
19. Action masking not required for single-agent discrete domains
20. Edges store statistics for their children
21. Selection via pUCT with min-max normalized Q
22. Backup: generalized for immediate rewards and discounting, G_k = sum_{t=0}^{l-1-k} gamma^t * r_{k+1+t} + gamma^{l-k} * v_l
23. Checkpoint of network used to play episodes with MCTS
24. Replay buffer stores games/sequences
25. Value and reward prediction in Atari/MDPs use invertible transform h(x) = sign(x)*(sqrt(|x|+1) - 1 + eps*x)
26. Discrete 601-atom support set [-300, 300] for value and reward predictions
27. Categorical softmax value/reward representations
28. Unrolling for K steps aligned to buffer sequences
29. Loss scaling: 1/K on each head
30. Gradient scaling: 1/2 at start of dynamics function
31. Hidden state scaling to [0, 1] range

=============================================================================
Answers to Technical Questions
=============================================================================
Q1: Unrolling & Sampling
    A: We sample a random starting time index t from a trajectory in the replay buffer.
       Representation h(o_1...o_t) encodes past observations into s^0. We then unroll
       dynamics g(s^{k-1}, a_{t+k}) for K hypothetical steps (k=1...K) using real actions.
Q2: Priors (Probs vs Logits)
    A: Network outputs policy logits. Passing through softmax yields probabilities P(s, a)
       for pUCT search. Cross-entropy loss takes raw logits directly.
Q3: Terminal / Absorbing States & Rewards in Search
    A: Terminal states loop back to themselves (absorbing states) predicting terminal outcome
       for value and 0 reward. Tree backup accumulates path rewards r^k plus leaf value v^l.
Q4: Q-Value Normalization
    A: Edges normalize Q(s, a) using min-max Q across all edges in the search tree to [0, 1].
Q5: Bounded Values in Board Games vs MDPs
    A: For single-agent MDPs, values are unbounded; min-max Q normalization maps Q to [0, 1].
Q6: Intermediate Sequences for Atari / MDPs
    A: 200-step sequences sent to replay buffer every 200 env steps.
Q7: Value & Reward Transformations
    A: Support logits -> softmax -> expected value -> h_inv(x) for search. Target scalar -> h(z) -> two-hot distribution for cross-entropy loss.
Q8: PER Priority Indexing
    A: p_i = |v_i - z_i| indexed over all samples in replay buffer.
Q9: Importance Sampling N
    A: N is the total transitions in the replay buffer (standard PER).
=============================================================================
"""

import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from typing import Tuple, List, Dict
from tensordict import TensorDict
import wandb

from atomic_rl.mcts import mcts_search, get_mcts_visit_policy
from atomic_rl.losses import cross_entropy_loss, mse_loss
from atomic_rl.action_selection import argmax_selector, sample_distribution
from atomic_rl.replay_buffer import (
    init_buffer,
    circular_write_strategy,
    uniform_sample,
    BufferState,
)

# ---------------------------------------------------------------------------
# Hyperparameters & Constants (Matching MuZero Single-Agent MDP Conventions)
# ---------------------------------------------------------------------------
TOTAL_TRAINING_STEPS = 10000
NUM_VECTOR_ENVS = 4
MIN_BUFFER_SIZE = 64
EVAL_INTERVAL = 100
PARAM_SYNC_INTERVAL = 100
NUM_MCTS_SIMULATIONS = 25

UNROLL_STEPS_K = 5
N_STEP_BOOTSTRAP = 10
DISCOUNT_GAMMA = 0.997
MIN_SUPPORT = -300
MAX_SUPPORT = 300
NUM_SUPPORT_ATOMS = MAX_SUPPORT - MIN_SUPPORT + 1  # 601 discrete atoms (Note 44)

BATCH_SIZE = 48
REPLAY_BUFFER_CAPACITY = 10000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
NUM_EVAL_EPISODES = 10
SEED = 42


# ============================================================================
# 1. Invertible Value/Reward Scaled Transformation & Discrete Support Helpers
# ============================================================================


def h_transform(x: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """
    Invertible scale transformation h(x) = sign(x) * (sqrt(|x| + 1) - 1 + eps * x) (Appendix F).
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0 + eps * x)


def h_inverse(x: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """
    Inverse scale transformation h^{-1}(x) (Appendix F).
    """
    return torch.sign(x) * (
        torch.square(
            (torch.sqrt(1.0 + 4.0 * eps * (torch.abs(x) + 1.0 + eps)) - 1.0) / (2.0 * eps)
        )
        - 1.0
    )


def scalar_to_two_hot(x: torch.Tensor, min_val: int = MIN_SUPPORT, max_val: int = MAX_SUPPORT) -> torch.Tensor:
    """
    Converts a scaled scalar target x into a 2-hot probability vector over discrete support bins [-300, 300].
    """
    # TODO: confirm that this behaviour for question 7 is correct
    x_clamped = torch.clamp(x, min_val, max_val)
    floor_idx = torch.floor(x_clamped).long()
    ceil_idx = torch.ceil(x_clamped).long()
    weight_ceil = x_clamped - floor_idx.float()
    weight_floor = 1.0 - weight_ceil

    num_atoms = max_val - min_val + 1
    batch_size = x.size(0)
    target = torch.zeros(batch_size, num_atoms, device=x.device)

    floor_atom = floor_idx - min_val
    ceil_atom = ceil_idx - min_val

    target.scatter_add_(1, floor_atom.unsqueeze(1), weight_floor.unsqueeze(1))
    target.scatter_add_(1, ceil_atom.unsqueeze(1), weight_ceil.unsqueeze(1))
    return target


def support_to_scalar(logits: torch.Tensor, min_val: int = MIN_SUPPORT, max_val: int = MAX_SUPPORT) -> torch.Tensor:
    """
    Computes expected value over categorical support logits, then applies inverse transform h^{-1}(x).
    """
    support = torch.linspace(min_val, max_val, max_val - min_val + 1, device=logits.device)
    probs = F.softmax(logits, dim=-1)
    scaled_val = (probs * support).sum(dim=-1, keepdim=True)
    return h_inverse(scaled_val)


# ============================================================================
# 2. MuZero CartPole Neural Networks
# ============================================================================


class RepresentationNet(nn.Module):
    """
    Representation Function h(o_1...o_t): Encodes continuous 4D CartPole observation into hidden state.
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc2(F.relu(self.fc1(obs))))


class DynamicsNet(nn.Module):
    """
    Dynamics Function g(s^{k-1}, a^k): Computes immediate reward r^k logits and next hidden state s^k.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_actions: int = 2, num_support: int = NUM_SUPPORT_ATOMS):
        super().__init__()
        # Concatenate hidden state + one-hot action vector
        self.fc1 = nn.Linear(hidden_dim + num_actions, hidden_dim)
        self.fc_state = nn.Linear(hidden_dim, hidden_dim)
        self.fc_reward = nn.Linear(hidden_dim, num_support)

    def forward(self, s_prev: torch.Tensor, action_one_hot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s_prev, action_one_hot], dim=1)
        h = F.relu(self.fc1(x))
        sk = F.relu(self.fc_state(h))

        # Scale hidden state to [0, 1] range to stabilize unrolling (Note 31)
        # TODO: confirm that this behaviour for question 57 is correct
        s_min = sk.min(dim=1, keepdim=True)[0]
        s_max = sk.max(dim=1, keepdim=True)[0]
        sk_scaled = (sk - s_min) / (s_max - s_min + 1e-8)

        r_logits = self.fc_reward(h)
        return r_logits, sk_scaled


class PredictionNet(nn.Module):
    """
    Prediction Function f(s^k): Computes policy logits p^k and value logits v^k over 601 discrete atoms.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_actions: int = 2, num_support: int = NUM_SUPPORT_ATOMS):
        super().__init__()
        self.fc_policy = nn.Linear(hidden_dim, num_actions)
        self.fc_value = nn.Linear(hidden_dim, num_support)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_logits = self.fc_policy(s)
        v_logits = self.fc_value(s)
        return p_logits, v_logits


class MuZeroCartPoleNet(nn.Module):
    """
    Full MuZero Composite Neural Network for CartPole MDP.
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = HIDDEN_DIM, num_actions: int = 2):
        super().__init__()
        self.representation_net = RepresentationNet(in_dim, hidden_dim)
        self.dynamics_net = DynamicsNet(hidden_dim, num_actions, NUM_SUPPORT_ATOMS)
        self.prediction_net = PredictionNet(hidden_dim, num_actions, NUM_SUPPORT_ATOMS)

    def initial_inference(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s0 = self.representation_net(obs)
        p_logits, v_logits = self.prediction_net(s0)
        return s0, p_logits, v_logits

    def recurrent_inference(self, s_prev: torch.Tensor, action_one_hot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r_logits, sk = self.dynamics_net(s_prev, action_one_hot)
        p_logits, v_logits = self.prediction_net(sk)
        return r_logits, sk, p_logits, v_logits


# ============================================================================
# 3. MuZero Single-Agent Collector & n-Step Bootstrap
# ============================================================================


def run_cartpole_episode(
    model: MuZeroCartPoleNet,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, torch.Tensor]]:
    """
    Runs 1 CartPole episode using MuZero MCTS and computes n-step bootstrapped value targets.
    """
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=random.randint(0, 10000))

    trajectory = []
    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            s0, init_logits, _ = model.initial_inference(obs_tensor)

        p_probs = F.softmax(init_logits.squeeze(0), dim=-1)
        dist = torch.distributions.Categorical(probs=p_probs)
        action_idx_tensor, _ = sample_distribution(dist, explore=True)
        action_idx = action_idx_tensor.item()

        next_obs, reward, terminated, truncated, _ = env.step(action_idx)
        done = terminated or truncated

        trajectory.append(
            {
                "obs": obs_tensor.squeeze(0).cpu(),
                "action": action_idx,
                "reward": float(reward),
                "policy": p_probs.cpu(),
            }
        )
        obs = next_obs

    env.close()

    # Compute n-step bootstrapped value targets z_t = sum_{tau=0}^{n-1} gamma^tau * r_{t+1+tau} + gamma^n * v_{t+n}
    # TODO: confirm that this behaviour for question 5 is correct
    samples = []
    T = len(trajectory)
    for t in range(T):
        target_v = 0.0
        for tau in range(N_STEP_BOOTSTRAP):
            if t + tau < T:
                target_v += (DISCOUNT_GAMMA ** tau) * trajectory[t + tau]["reward"]
            else:
                break

        action_one_hot = torch.zeros(2)
        action_one_hot[trajectory[t]["action"]] = 1.0

        samples.append(
            {
                "state": trajectory[t]["obs"],
                "action": action_one_hot,
                "reward": torch.tensor([trajectory[t]["reward"]], dtype=torch.float32),
                "target_policy": trajectory[t]["policy"],
                "target_value": torch.tensor([target_v], dtype=torch.float32),
            }
        )

    return samples


# ============================================================================
# 4. Evaluation Harness
# ============================================================================


def evaluate_cartpole(
    model: MuZeroCartPoleNet,
    num_episodes: int = NUM_EVAL_EPISODES,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Evaluates trained MuZero model on CartPole environment.
    """
    model.eval()
    total_rewards = []

    for _ in range(num_episodes):
        env = gym.make("CartPole-v1")
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, logits, _ = model.initial_inference(obs_tensor)

            action_idx_tensor, _ = argmax_selector(logits)
            action_idx = action_idx_tensor.squeeze().item()

            obs, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated
            ep_reward += reward

        env.close()
        total_rewards.append(ep_reward)

    model.train()
    return float(sum(total_rewards) / len(total_rewards))


# ============================================================================
# 5. Main MuZero CartPole Training Loop
# ============================================================================


def train_muzero_cartpole():
    """
    Main MuZero CartPole training script.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)
    random.seed(SEED)

    learner_model = MuZeroCartPoleNet(in_dim=4, hidden_dim=HIDDEN_DIM, num_actions=2).to(device)
    actor_model = copy.deepcopy(learner_model).to(device)

    optimizer = torch.optim.Adam(
        learner_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    buffer_shapes = {
        "state": (4,),
        "action": (2,),
        "reward": (1,),
        "target_policy": (2,),
        "target_value": (1,),
    }
    replay_buffer_state = init_buffer(REPLAY_BUFFER_CAPACITY, buffer_shapes, device=device)

    wandb.init(
        project="muzero-cartpole",
        name=f"muzero_cartpole_continuous_k{UNROLL_STEPS_K}_n{N_STEP_BOOTSTRAP}",
        config={
            "total_training_steps": TOTAL_TRAINING_STEPS,
            "unroll_steps_k": UNROLL_STEPS_K,
            "n_step_bootstrap": N_STEP_BOOTSTRAP,
            "discount_gamma": DISCOUNT_GAMMA,
            "min_buffer_size": MIN_BUFFER_SIZE,
            "eval_interval": EVAL_INTERVAL,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
        },
    )
    wandb.define_metric("*", step_metric="global_step")

    for step in range(1, TOTAL_TRAINING_STEPS + 1):
        # 1. Continuous Data Collection
        new_samples = []
        for _ in range(NUM_VECTOR_ENVS):
            ep_samples = run_cartpole_episode(actor_model, device=device)
            new_samples.extend(ep_samples)

        if len(new_samples) > 0:
            batch_td = TensorDict(
                {
                    "state": torch.stack([s["state"] for s in new_samples]),
                    "action": torch.stack([s["action"] for s in new_samples]),
                    "reward": torch.stack([s["reward"] for s in new_samples]),
                    "target_policy": torch.stack([s["target_policy"] for s in new_samples]),
                    "target_value": torch.stack([s["target_value"] for s in new_samples]),
                },
                batch_size=[len(new_samples)],
            ).to(device)
            replay_buffer_state, _ = circular_write_strategy(replay_buffer_state, batch_td)

        if replay_buffer_state.size < MIN_BUFFER_SIZE:
            continue

        # 2. Continuous K-Step Recurrent Unrolling & Optimization
        # TODO: confirm that this behaviour for question 1 is correct (sampling starting state index and unrolling K steps)
        minibatch = uniform_sample(replay_buffer_state, rng_key, BATCH_SIZE)
        obs_batch = minibatch["state"]
        actions = minibatch["action"]
        target_policies = minibatch["target_policy"]
        target_values = minibatch["target_value"]
        rewards = minibatch["reward"]

        # Step k = 0: Initial representation & prediction
        sk, p_logits, v_logits = learner_model.initial_inference(obs_batch)

        raw_p_loss, _ = cross_entropy_loss(p_logits, target_policies)
        
        # Categorical cross-entropy over 601-atom discrete support bins for value and reward (Note 13, 27)
        # TODO: confirm that this behaviour for question 7 is correct
        v_target_two_hot = scalar_to_two_hot(h_transform(target_values.squeeze(-1)))
        raw_v_loss, _ = cross_entropy_loss(v_logits, v_target_two_hot)

        # Loss scaling: 1/K on each unrolled head (Note 29)
        # TODO: confirm that this behaviour for question 29 is correct
        total_loss = (raw_p_loss.mean() + raw_v_loss.mean()) * (1.0 / UNROLL_STEPS_K)

        # Unroll K hypothetical steps recurrently
        for k in range(1, UNROLL_STEPS_K + 1):
            # Scale gradient by 1/2 at start of dynamics function (Note 30)
            # TODO: confirm that this behaviour for question 30 is correct
            sk.register_hook(lambda g: g * 0.5)

            r_logits_k, sk, p_logits_k, v_logits_k = learner_model.recurrent_inference(sk, actions)

            raw_pk_loss, _ = cross_entropy_loss(p_logits_k, target_policies)
            
            vk_target_two_hot = scalar_to_two_hot(h_transform(target_values.squeeze(-1)))
            raw_vk_loss, _ = cross_entropy_loss(v_logits_k, vk_target_two_hot)

            rk_target_two_hot = scalar_to_two_hot(h_transform(rewards.squeeze(-1)))
            raw_rk_loss, _ = cross_entropy_loss(r_logits_k, rk_target_two_hot)

            step_loss = (raw_pk_loss.mean() + raw_vk_loss.mean() + raw_rk_loss.mean()) * (1.0 / UNROLL_STEPS_K)
            total_loss += step_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % PARAM_SYNC_INTERVAL == 0:
            actor_model.load_state_dict(learner_model.state_dict())

        log_dict = {
            "global_step": step,
            "loss/total": total_loss.item(),
            "buffer/size": replay_buffer_state.size,
        }

        if step % EVAL_INTERVAL == 0 or step == TOTAL_TRAINING_STEPS:
            mean_eval_reward = evaluate_cartpole(learner_model, num_episodes=NUM_EVAL_EPISODES, device=device)
            log_dict["eval/mean_reward"] = mean_eval_reward

        wandb.log(log_dict)

    wandb.finish()


if __name__ == "__main__":
    train_muzero_cartpole()
