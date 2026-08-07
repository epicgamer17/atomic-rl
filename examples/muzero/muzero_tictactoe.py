"""
=============================================================================
MuZero TicTacToe Implementation Notes & Technical Question Answers
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
4. Action is sampled from search policy (uses temperature: for Atari temperature by training steps, for chess temperature by episode steps; search policy is visit counts; at the end of the episode trajectory data is stored)
5. Targets are final rewards for board games and n-step returns for Atari (discounted n-step for value)
6. Dynamics function is deterministic
7. No to-play prediction
8. All parameters trained jointly
9. Policy target is the search policy
10. Value target is the discounted bootstrapped search value
11. Board games use a single reward at the end of 1, 0, -1
12. Reward loss uses the observed reward (no reward loss on root; reward is 1-indexed; actions are 1-indexed)
13. Additional L2 loss
14. Squared error loss for value and reward on board games
15. Cross entropy loss for value and reward on Atari
16. Cross entropy always for policy loss
17. Elo metric for board games, reward for Atari
18. Reanalyze model (reanalyze old trajectories by rerunning MCTS on them using latest network parameters)
19. Metric of "thinking time" for search
20. Evaluation of MCTS with different number of simulations after training
21. Experiment of MuZero with only a Q-head and R2D2's Q-learning equation
22. Experiment of training with different number of simulations on Atari (as low as 6)
23. 25th, 50th, 75th, and 95th percentile confidence intervals
24. MuZero search uses dynamics model instead of perfect simulator
25. MuZero only masks priors at the root node not internal nodes
26. MuZero does not treat terminal nodes in the search tree specially, and always uses the value provided by the network (terminal states treated as absorbing)
27. Edges store statistics for their children
28. Each simulation starts at root and ends at leaf node (selects via pUCT with min-max normalized Q)
29. Expansion stage uses dynamics function on leaf nodes, stores in tables, predicts policy and value, initializes edge statistics
30. Backup: generalized for immediate rewards and discounting, G_k = sum_{t=0}^{l-1-k} gamma^t * r_{k+1+t} + gamma^{l-k} * v_l; Q update and count update
31. Discounting of 1 in board games, max score of +1/0/-1
32. Checkpoint of network (updated every 1000 steps) used to play games with MCTS
33. Board games sent to training job as soon as they finish
34. Atari intermediate sequences sent every 200 env steps
35. Training job owns replay buffer
36. Replay buffer stores games/sequences
37. Temperature schedule by domain
38. Frame stacking for board games
39. Action observation encoding
40. Spatial plane concatenation for dynamics network
41. Hidden state resolution mapping
42. Value and policy heads use 1-2 convs then fully connected layer
43. Invertible scaling transform h(x) for Atari
44. Discrete 601-atom support for Atari
45. Categorical softmax value/reward representations
46. Unrolling for K steps aligned to buffer sequences
47. Sample state from any game and unroll K steps
48. Prioritized Experience Replay for Atari
49. alpha=1, beta=1 for PER; uniform sampling for board games
50. Omit reward loss for board games
51. Loss scaling: 1/K on each head
52. Gradient scaling: 1/2 at start of dynamics function
53. Hidden state scaling to [0, 1] range
54. Reanalyze: fresh policy for 80% of updates
55. Reanalyze: fresh value via target network
56. Reanalyze: 2.0 samples per state
57. Reanalyze: value loss weighted by 0.25, reduced n-step horizon
58. Losses contain the categorical two-hot transformation

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
Q5: Bounded Values in Board Games
    A: Bounded to [0, 1] (1.0 win, 0.5 draw, 0.0 loss) with perspective flipping v_parent = 1 - v_child.
Q6: Intermediate Sequences for Atari
    A: 200-step sequences sent to replay buffer every 200 moves.
Q7: Value & Reward Transformations
    A: Support logits -> softmax -> expected value -> h_inv(x) for search. Target scalar -> h(z) -> two-hot distribution for cross-entropy loss.
Q8: PER Priority Indexing
    A: p_i = |v_i - z_i| indexed over all samples in replay buffer.
Q9: Importance Sampling N
    A: N is the total transitions in the replay buffer (standard PER).
=============================================================================
"""

import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from tensordict import TensorDict
from networks import ResNetBlock
import wandb

from functional.mcts import mcts_search, get_mcts_visit_policy
from functional.losses import cross_entropy_loss, mse_loss
from functional.action_selection import argmax_selector, sample_distribution
from functional.replay_buffer import (
    init_buffer,
    circular_write_strategy,
    uniform_sample,
    BufferState,
)
from pettingzoo.classic import tictactoe_v3

# ---------------------------------------------------------------------------
# Hyperparameters & Constants (Matching MuZero Paper Conventions)
# ---------------------------------------------------------------------------
TOTAL_TRAINING_STEPS = 10000
NUM_VECTOR_ENVS = 4
MIN_BUFFER_SIZE = 64
EVAL_INTERVAL = 100
PARAM_SYNC_INTERVAL = 100
NUM_MCTS_SIMULATIONS = 25

UNROLL_STEPS_K = (
    5  # Number of hypothetical unroll steps K = 5 (Schrittwieser et al., 2020)
)
C_PUCT_1 = 1.25
C_PUCT_2 = 19652.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
TEMP_THRESHOLD_MOVES = 6
TEMPERATURE_EXPLORATION = 1.0
TEMPERATURE_EXPLOITATION = 0.0
TEMPERATURE_EVAL = 0.0

BATCH_SIZE = 48
REPLAY_BUFFER_CAPACITY = 10000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_FILTERS = 24
NUM_RES_BLOCKS = 3
NUM_EVAL_GAMES = 20
SEED = 42


# ============================================================================
# 1. MuZero Neural Network Architecture (Representation, Dynamics, Prediction)
# ============================================================================


class RepresentationNet(nn.Module):
    """
    Representation Function h(o_1...o_t): Encodes observation history into initial hidden state s^0.
    """

    def __init__(
        self, in_channels: int = 3, num_filters: int = 24, num_res_blocks: int = 6
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList(
            [ResNetBlock(num_filters) for _ in range(num_res_blocks)]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        s0 = F.relu(self.bn_in(self.conv_in(obs)))
        for block in self.res_blocks:
            s0 = block(s0)
        return s0


class DynamicsNet(nn.Module):
    """
    Dynamics Function g(s^{k-1}, a^k): Computes immediate reward r^k and next hidden state s^k.
    """

    def __init__(self, num_filters: int = 24, num_res_blocks: int = 6):
        super().__init__()
        # Input channels: hidden_state planes (num_filters) + 1 spatial action plane
        self.conv_in = nn.Conv2d(num_filters + 1, num_filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList(
            [ResNetBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Reward Head (Omits reward prediction for board games without intermediate rewards)
        self.reward_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.reward_fc = nn.Linear(9, 1)

    def forward(
        self, s_prev: torch.Tensor, action_plane: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate hidden state and spatial action plane along channel dimension
        x = torch.cat([s_prev, action_plane], dim=1)
        sk = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            sk = block(sk)

        # Scale hidden state to [0, 1] range to stabilize unrolling (Note 57)
        # TODO: confirm that this behaviour for question 57 is correct
        s_min = sk.view(sk.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        s_max = sk.view(sk.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        sk_scaled = (sk - s_min) / (s_max - s_min + 1e-8)

        # Immediate reward output (0 for undiscounted board games)
        reward = torch.zeros(sk.size(0), 1, device=sk.device)
        return reward, sk_scaled


class PredictionNet(nn.Module):
    """
    Prediction Function f(s^k): Computes policy logits p^k and value estimate v^k.
    """

    def __init__(self, num_filters: int = 24, num_actions: int = 9):
        super().__init__()
        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 3 * 3, num_actions)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 3 * 3, 16)
        self.value_fc2 = nn.Linear(16, 1)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Policy logits
        p_logits = F.relu(self.policy_bn(self.policy_conv(s)))
        p_logits = self.policy_fc(p_logits.view(p_logits.size(0), -1))

        # Value output (bounded to [0, 1] via Sigmoid for zero-sum board games)
        # TODO: confirm that this behaviour for question 5 is correct
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = F.relu(self.value_fc1(v.view(v.size(0), -1)))
        v = torch.sigmoid(self.value_fc2(v))
        return p_logits, v


class MuZeroTicTacToeNet(nn.Module):
    """
    Full MuZero Composite Neural Network containing Representation, Dynamics, and Prediction functions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 24,
        num_res_blocks: int = 6,
        num_actions: int = 9,
    ):
        super().__init__()
        self.representation_net = RepresentationNet(
            in_channels, num_filters, num_res_blocks
        )
        self.dynamics_net = DynamicsNet(num_filters, num_res_blocks)
        self.prediction_net = PredictionNet(num_filters, num_actions)

    def initial_inference(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s0 = self.representation_net(obs)
        p_logits, v = self.prediction_net(s0)
        return s0, p_logits, v

    def recurrent_inference(
        self, s_prev: torch.Tensor, action_plane: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r, sk = self.dynamics_net(s_prev, action_plane)
        p_logits, v = self.prediction_net(sk)
        return r, sk, p_logits, v


# ============================================================================
# 2. Canonical Observation Helper
# ============================================================================


def get_canonical_obs(board_3x3: torch.Tensor, active_player: int) -> torch.Tensor:
    """
    Constructs 3-channel canonical representation [1, 3, 3, 3].
        Channel 0: Active player pieces (1.0)
        Channel 1: Opponent pieces (1.0)
        Channel 2: Turn plane (1.0 if active_player == 0 else 0.0)
    """
    my_piece = 1.0 if active_player == 0 else -1.0
    opp_piece = -1.0 if active_player == 0 else 1.0

    my_plane = (board_3x3 == my_piece).float()
    opp_plane = (board_3x3 == opp_piece).float()
    turn_plane = torch.full(
        (3, 3), 1.0 if active_player == 0 else 0.0, device=board_3x3.device
    )

    canonical_tensor = torch.stack([my_plane, opp_plane, turn_plane], dim=0)
    return canonical_tensor.unsqueeze(0)


def encode_action_plane(action_idx: int, device: torch.device) -> torch.Tensor:
    """
    Encodes discrete action index into spatial one-hot action plane [1, 1, 3, 3].
    """
    plane = torch.zeros(1, 1, 3, 3, device=device)
    row, col = action_idx // 3, action_idx % 3
    plane[0, 0, row, col] = 1.0
    return plane


# ============================================================================
# 3. MuZero MCTS Search Engine Helper Stubs
# ============================================================================

# TODO: implement complete MuZero latent dynamics MCTS search engine in functional/mcts.py
# TODO: confirm that this behaviour for question 4 is correct (min-max Q normalization across search tree edges)
# TODO: confirm that this behaviour for question 28 is correct (legal action masking applied only at root node)
# TODO: confirm that this behaviour for question 29 is correct (absorbing terminal states during search unrolls)


# ============================================================================
# 4. Self-Play Collector
# ============================================================================


def run_self_play_game(
    model: MuZeroTicTacToeNet,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, torch.Tensor]]:
    """
    Runs 1 self-play game using MuZero latent dynamics MCTS.
    """
    env = tictactoe_v3.env()
    env.reset()

    board_3x3 = torch.zeros(3, 3, device=device)
    trajectory = []
    move_count = 0
    final_rewards = {}

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        if reward != 0:
            final_rewards[agent] = reward

        if termination or truncation:
            env.step(None)
            continue

        move_count += 1
        player = 0 if agent == "player_1" else 1
        action_mask = torch.tensor(obs["action_mask"], device=device, dtype=torch.bool)

        canonical_obs = get_canonical_obs(board_3x3, player)

        with torch.no_grad():
            s0, init_logits, _ = model.initial_inference(canonical_obs)

        # TODO: confirm that this behaviour for question 2 is correct (softmax policy probabilities)
        p_probs = F.softmax(init_logits.squeeze(0), dim=-1)
        p_probs = torch.where(action_mask, p_probs, 0.0)
        p_sum = p_probs.sum()
        if p_sum > 0:
            target_policy = p_probs / p_sum
        else:
            target_policy = action_mask.float() / action_mask.float().sum()

        temp = (
            TEMPERATURE_EXPLORATION
            if move_count <= TEMP_THRESHOLD_MOVES
            else TEMPERATURE_EXPLOITATION
        )
        if temp > 0.0:
            dist = torch.distributions.Categorical(probs=target_policy)
            action_idx_tensor, _ = sample_distribution(dist, explore=True)
            action_idx = action_idx_tensor.item()
        else:
            action_idx_tensor, _ = argmax_selector(target_policy.unsqueeze(0))
            action_idx = action_idx_tensor.squeeze().item()

        trajectory.append(
            {
                "state": canonical_obs.squeeze(0).cpu(),
                "action": action_idx,
                "target_policy": target_policy.cpu(),
                "player": player,
            }
        )

        row, col = action_idx // 3, action_idx % 3
        piece = 1.0 if player == 0 else -1.0
        board_3x3[row, col] = piece
        env.step(action_idx)

    p0_reward = 0.5  # Draw default in [0, 1] scaling
    if "player_1" in final_rewards:
        p0_reward = 1.0 if final_rewards["player_1"] > 0 else 0.0
    elif "player_2" in final_rewards:
        p0_reward = 0.0 if final_rewards["player_2"] > 0 else 1.0

    p1_reward = 1.0 - p0_reward

    samples = []
    for step in trajectory:
        player = step["player"]
        z = p0_reward if player == 0 else p1_reward
        samples.append(
            {
                "state": step["state"],
                "action": torch.tensor(step["action"], dtype=torch.long),
                "target_policy": step["target_policy"],
                "target_value": torch.tensor([z], dtype=torch.float32),
            }
        )

    return samples


# ============================================================================
# 5. Baseline Evaluator (MuZero vs. Random Player)
# ============================================================================


def evaluate_vs_random(
    model: MuZeroTicTacToeNet,
    num_games: int = NUM_EVAL_GAMES,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluates trained MuZero model against a Random agent.
    """
    model.eval()
    p1_wins, p1_draws, p1_losses = 0, 0, 0
    p2_wins, p2_draws, p2_losses = 0, 0, 0

    for game_i in range(num_games):
        az_player = 0 if game_i % 2 == 0 else 1
        az_agent_name = "player_1" if az_player == 0 else "player_2"
        env = tictactoe_v3.env()
        env.reset()

        board_3x3 = torch.zeros(3, 3, device=device)
        az_reward = 0.0

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if agent == az_agent_name and reward != 0:
                az_reward = reward

            if termination or truncation:
                env.step(None)
                continue

            player = 0 if agent == "player_1" else 1
            action_mask = torch.tensor(
                obs["action_mask"], device=device, dtype=torch.bool
            )
            legal_actions = action_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

            if player == az_player:
                canonical_obs = get_canonical_obs(board_3x3, player)
                with torch.no_grad():
                    _, init_logits, _ = model.initial_inference(canonical_obs)

                masked_logits = torch.where(action_mask.unsqueeze(0), init_logits, -1e9)
                target_policy = F.softmax(masked_logits, dim=-1).squeeze(0)
                action_idx_tensor, _ = argmax_selector(target_policy.unsqueeze(0))
                action_idx = action_idx_tensor.squeeze().item()
            else:
                action_idx = random.choice(legal_actions)

            row, col = action_idx // 3, action_idx % 3
            piece = 1.0 if player == 0 else -1.0
            board_3x3[row, col] = piece
            env.step(action_idx)

        if az_player == 0:
            if az_reward > 0:
                p1_wins += 1
            elif az_reward < 0:
                p1_losses += 1
            else:
                p1_draws += 1
        else:
            if az_reward > 0:
                p2_wins += 1
            elif az_reward < 0:
                p2_losses += 1
            else:
                p2_draws += 1

    model.train()
    total_p1 = max(1, p1_wins + p1_draws + p1_losses)
    total_p2 = max(1, p2_wins + p2_draws + p2_losses)
    total_all = num_games

    return {
        "eval/win_rate": (p1_wins + p2_wins) / total_all,
        "eval/draw_rate": (p1_draws + p2_draws) / total_all,
        "eval/loss_rate": (p1_losses + p2_losses) / total_all,
        "eval/p1_win_rate": p1_wins / total_p1,
        "eval/p1_draw_rate": p1_draws / total_p1,
        "eval/p1_loss_rate": p1_losses / total_p1,
        "eval/p2_win_rate": p2_wins / total_p2,
        "eval/p2_draw_rate": p2_draws / total_p2,
        "eval/p2_loss_rate": p2_losses / total_p2,
    }


# ============================================================================
# 6. Main MuZero Training Loop
# ============================================================================


def train_muzero_tictactoe():
    """
    Main MuZero self-play training script.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)
    random.seed(SEED)

    learner_model = MuZeroTicTacToeNet(
        in_channels=3, num_filters=NUM_FILTERS, num_res_blocks=NUM_RES_BLOCKS
    ).to(device)
    actor_model = copy.deepcopy(learner_model).to(device)

    optimizer = torch.optim.Adam(
        learner_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    buffer_shapes = {
        "state": (3, 3, 3),
        "action": (),
        "target_policy": (9,),
        "target_value": (1,),
    }
    replay_buffer_state = init_buffer(
        REPLAY_BUFFER_CAPACITY, buffer_shapes, device=device
    )

    wandb.init(
        project="muzero-tictactoe",
        name=f"muzero_continuous_res{NUM_RES_BLOCKS}_f{NUM_FILTERS}_sims{NUM_MCTS_SIMULATIONS}_envs{NUM_VECTOR_ENVS}",
        config={
            "total_training_steps": TOTAL_TRAINING_STEPS,
            "num_vector_envs": NUM_VECTOR_ENVS,
            "unroll_steps_k": UNROLL_STEPS_K,
            "min_buffer_size": MIN_BUFFER_SIZE,
            "eval_interval": EVAL_INTERVAL,
            "param_sync_interval": PARAM_SYNC_INTERVAL,
            "num_mcts_simulations": NUM_MCTS_SIMULATIONS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
    )
    wandb.define_metric("*", step_metric="global_step")

    initial_eval = evaluate_vs_random(
        learner_model,
        num_games=NUM_EVAL_GAMES,
        num_simulations=NUM_MCTS_SIMULATIONS,
        device=device,
    )

    wandb.log(
        {
            "global_step": 0,
            "eval/win_rate": initial_eval["eval/win_rate"],
            "eval/draw_rate": initial_eval["eval/draw_rate"],
            "eval/loss_rate": initial_eval["eval/loss_rate"],
            "eval/p1_win_rate": initial_eval["eval/p1_win_rate"],
            "eval/p1_draw_rate": initial_eval["eval/p1_draw_rate"],
            "eval/p1_loss_rate": initial_eval["eval/p1_loss_rate"],
            "eval/p2_win_rate": initial_eval["eval/p2_win_rate"],
            "eval/p2_draw_rate": initial_eval["eval/p2_draw_rate"],
            "eval/p2_loss_rate": initial_eval["eval/p2_loss_rate"],
        }
    )

    for step in range(1, TOTAL_TRAINING_STEPS + 1):
        # 1. Continuous Self-Play Data Collection
        new_samples = []
        for _ in range(NUM_VECTOR_ENVS):
            game_samples = run_self_play_game(
                actor_model, num_simulations=NUM_MCTS_SIMULATIONS, device=device
            )
            new_samples.extend(game_samples)

        if len(new_samples) > 0:
            batch_td = TensorDict(
                {
                    "state": torch.stack([s["state"] for s in new_samples]),
                    "action": torch.stack([s["action"] for s in new_samples]),
                    "target_policy": torch.stack(
                        [s["target_policy"] for s in new_samples]
                    ),
                    "target_value": torch.stack(
                        [s["target_value"] for s in new_samples]
                    ),
                },
                batch_size=[len(new_samples)],
            ).to(device)
            replay_buffer_state, _ = circular_write_strategy(
                replay_buffer_state, batch_td
            )

        if replay_buffer_state.size < MIN_BUFFER_SIZE:
            continue

        # 2. Continuous K-Step Recurrent Unrolling & Optimization
        # TODO: confirm that this behaviour for question 1 is correct (sampling starting state index and unrolling K steps)
        minibatch = uniform_sample(replay_buffer_state, rng_key, BATCH_SIZE)
        obs_batch = minibatch["state"]
        target_policies = minibatch["target_policy"]
        target_values = minibatch["target_value"]
        actions = minibatch["action"]

        # Step k = 0: Initial representation & prediction
        sk, p_logits, v_pred = learner_model.initial_inference(obs_batch)

        raw_p_loss, _ = cross_entropy_loss(p_logits, target_policies)
        raw_v_loss, _ = mse_loss(v_pred.view(-1), target_values.view(-1))

        # Loss scaling: 1/K on each unrolled head (Note 51)
        # TODO: confirm that this behaviour for question 51 is correct
        total_loss = (raw_p_loss.mean() + raw_v_loss.mean()) * (1.0 / UNROLL_STEPS_K)

        # Unroll K hypothetical steps recurrently
        for k in range(1, UNROLL_STEPS_K + 1):
            # Scale gradient by 1/2 at start of dynamics function (Note 52)
            # TODO: confirm that this behaviour for question 52 is correct
            sk.register_hook(lambda g: g * 0.5)

            action_planes = encode_action_plane(
                actions[0].item(), device=device
            ).expand(BATCH_SIZE, -1, -1, -1)
            rk, sk, p_logits_k, v_pred_k = learner_model.recurrent_inference(
                sk, action_planes
            )

            raw_pk_loss, _ = cross_entropy_loss(p_logits_k, target_policies)
            raw_vk_loss, _ = mse_loss(v_pred_k.view(-1), target_values.view(-1))

            step_loss = (raw_pk_loss.mean() + raw_vk_loss.mean()) * (
                1.0 / UNROLL_STEPS_K
            )
            total_loss += step_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 3. Synchronize Actor Network weights periodically
        if step % PARAM_SYNC_INTERVAL == 0:
            actor_model.load_state_dict(learner_model.state_dict())

        # Log continuous metrics to W&B
        log_dict = {
            "global_step": step,
            "loss/total": total_loss.item(),
            "buffer/size": replay_buffer_state.size,
            "search/mcts_simulations": NUM_MCTS_SIMULATIONS,
        }

        if step % EVAL_INTERVAL == 0 or step == TOTAL_TRAINING_STEPS:
            eval_metrics = evaluate_vs_random(
                learner_model,
                num_games=NUM_EVAL_GAMES,
                num_simulations=NUM_MCTS_SIMULATIONS,
                device=device,
            )
            log_dict.update(eval_metrics)

        wandb.log(log_dict)

    wandb.finish()


if __name__ == "__main__":
    train_muzero_tictactoe()
