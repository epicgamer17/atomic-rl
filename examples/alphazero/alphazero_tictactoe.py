"""
AlphaZero on PettingZoo TicTacToe
==================================

Paper Reference:
    "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
    Silver et al., Science 2018 (arXiv 2017: https://arxiv.org/abs/1712.01815)
    "Mastering the Game of Go without Human Knowledge"
    Silver et al., Nature 2017 (AlphaGo Zero)

Algorithm Summary:
    AlphaZero is a general reinforcement learning algorithm for two-player zero-sum games
    that learns entirely through self-play, starting from random initial weights without
    any domain-specific human knowledge or demonstration data.

TODO: In my own words. This is AI Generated
Key Contributions & Ideas:
    1. Tabula Rasa Self-Play: The agent plays games against itself using Monte Carlo Tree Search (MCTS) guided by a single deep neural network. The neural network learns from the outcomes of its own self-play games.
    2. Dual-Head Neural Network f_theta(s) = (p, v): A single network takes board representations and outputs both a policy vector p (prior probabilities for all possible moves) and a scalar value prediction v in [-1, +1] estimating expected outcome from the current player's perspective.
    3. MCTS as Policy Improvement: MCTS search uses network predictions (p, v) to guide node selection (PUCT algorithm). The visit count distribution pi at the root after search acts as a strongly improved policy target compared to raw network priors p.
    4. Policy Iteration Loop:
       - Self-Play: Execute MCTS to generate games, storing (s_t, pi_t, z_t) tuples where z_t is the final outcome (-1, 0, +1) relative to player at turn t.
       - Network Optimization: Train (p, v) by minimizing combined cross-entropy policy loss and MSE value loss: L = (z - v)^2 - pi^T * log(p) + c||theta||^2.

Differences in this Implementation:
    - Environment: Uses PettingZoo's `tictactoe_v3` environment for evaluation and self-play.
    - Lightweight Dynamics: Inlines a fast 9-cell 3x3 board simulator for MCTS tree transitions,
      avoiding environment copy overhead during search.
    - All-in-one Self-Contained Example: Includes network, dynamics simulator, loss function,
      self-play collector, replay buffer, and baseline evaluation harness.

NOTE: Focus of the library (when it comes to search based algos) is not on AlphaZero-like algorithms, but MuZero-like ones. This is here as a stepping stone for people looking to understand MuZero better, but in general I encourage you to look into model learned algorithms (like Dreamerv3, MuZero, etc.) over model given ones.

TODO: some hyperparameter tuning. it works well, but still loses sometimes to a random bot, which i remember when i had muzero working on the older library never happened. I imagine alphazero should be better.
"""

import copy
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from tensordict import TensorDict
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
# Hyperparameters & Constants (Matching AlphaZero Paper Conventions)
# ---------------------------------------------------------------------------
# Self-Play & MCTS Simulation Parameters
TOTAL_TRAINING_STEPS = (
    10000  # Total continuous training steps (1 SGD step per training loop)
)
NUM_VECTOR_ENVS = 4  # Number of parallel vectorized self-play environments per step
MIN_BUFFER_SIZE = 64  # Warmup buffer size before SGD optimization begins
EVAL_INTERVAL = 100  # Evaluate vs Random agent every N training steps
PARAM_SYNC_INTERVAL = (
    100  # Sync actor network weights from learner network every N steps
)
NUM_MCTS_SIMULATIONS = 25

# MCTS PUCT Search Constants (Silver et al., 2017/2018)
C_PUCT = 1.25  # PUCT exploration coefficient c_puct = 1.25
DIRICHLET_ALPHA = 1.0  # Dirichlet noise alpha (0.3 for games with ~9 moves)
DIRICHLET_EPSILON = 0.25  # Exploration noise fraction epsilon = 0.25

# Temperature Schedule Constants (Silver et al. 2017/2018)
# tau = 1.0 for first TEMP_THRESHOLD_MOVES moves in self-play, then tau -> 0.0 (greedy)
TEMP_THRESHOLD_MOVES = 5  # First N moves use tau = 1.0, remaining moves use tau = 0.0
TEMPERATURE_EXPLORATION = 1.0  # Temperature tau = 1.0 for initial exploratory moves
TEMPERATURE_EXPLOITATION = 0.0  # Temperature tau = 0.0 (greedy) for remaining moves
TEMPERATURE_EVAL = 0.0  # Temperature tau = 0.0 (greedy) for evaluation games

# Optimization & Architecture Parameters
BATCH_SIZE = 48
REPLAY_BUFFER_CAPACITY = 10000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization weight decay c = 10^-4
NUM_FILTERS = 24  # 16 filters per ResNet block
NUM_RES_BLOCKS = 6

# Evaluation & Seed
NUM_EVAL_GAMES = 100
SEED = 42


# ============================================================================
# 1. Dual-Head AlphaZero ResNet Neural Network
# ============================================================================


class ResNetBlock(nn.Module):
    """
    Standard Residual Block matching AlphaZero paper (Silver et al. 2017/2018).
    Contains 2 Conv2d layers with BatchNorm and a skip-connection residual addition.
    """

    def __init__(self, num_filters: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class TicTacToeNet(nn.Module):
    """
    AlphaZero Dual-Head ResNet Architecture for TicTacToe.

    Input: [B, 3, 3, 3] canonical state representation:
           - Channel 0: Active player pieces (1.0 where active player has pieces)
           - Channel 1: Opponent pieces (1.0 where opponent has pieces)
           - Channel 2: Player turn encoding plane (1.0 for Player 0 / 'X', 0.0 for Player 1 / 'O')

    Outputs:
        - policy_logits: [B, 9] unnormalized move action logits
        - value: [B, 1] predicted state evaluation scalar in [-1, +1]
    """

    def __init__(
        self, num_filters: int = NUM_FILTERS, num_res_blocks: int = NUM_RES_BLOCKS
    ):
        super().__init__()
        # Initial Convolutional Block
        self.conv_in = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_filters)

        # Residual Tower (2-3 ResNet Blocks of 16 filters)
        self.res_blocks = nn.ModuleList(
            [ResNetBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy Head (AlphaZero paper specification): Conv(1x1, 2 filters) -> BN -> ReLU -> FC(9)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 3 * 3, 9),
        )

        # Value Head (AlphaZero paper specification): Conv(1x1, 1 filter) -> BN -> ReLU -> FC(16) -> ReLU -> FC(1) -> Tanh
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 3 * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            features = block(features)

        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


# ============================================================================
# 3. Fast Inlined TicTacToe Board Simulator for MCTS Search
# ============================================================================


def check_tictactoe_winner(board: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Checks 3x3 board tensor (+1 for P0, -1 for P1, 0 for empty) for win or terminal draw.

    Args:
        board: [B, 3, 3] board tensor (+1 for P0, -1 for P1, 0 for empty).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (winner_tensor [B], is_terminal [B])
            winner: +1 if P0 won, -1 if P1 won, 0 if draw or ongoing.
    """
    batch_size = board.shape[0]
    device = board.device

    rows = board.sum(dim=2)  # [B, 3]
    cols = board.sum(dim=1)  # [B, 3]
    diag1 = torch.stack([board[:, 0, 0], board[:, 1, 1], board[:, 2, 2]], dim=1).sum(
        dim=1, keepdim=True
    )
    diag2 = torch.stack([board[:, 0, 2], board[:, 1, 1], board[:, 2, 0]], dim=1).sum(
        dim=1, keepdim=True
    )

    lines = torch.cat([rows, cols, diag1, diag2], dim=1)  # [B, 8]

    p0_wins = (lines == 3).any(dim=1)
    p1_wins = (lines == -3).any(dim=1)
    board_full = (board != 0).all(dim=1).all(dim=1)

    winner = torch.zeros(batch_size, device=device)
    winner = torch.where(p0_wins, torch.tensor(1.0, device=device), winner)
    winner = torch.where(p1_wins, torch.tensor(-1.0, device=device), winner)

    is_terminal = p0_wins | p1_wins | board_full
    return winner, is_terminal


def tictactoe_dynamics_fn(
    embeddings: torch.Tensor, actions_taken: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transition dynamics for MCTS simulation.

    Args:
        embeddings: State tensor [B, 3, 3, 2] (board, to_play)
        actions_taken: Action indices [B] in 0..8.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - next_embeddings: [B, 3, 3, 2]
            - reward: [B] (terminal reward relative to current player)
            - next_to_play: [B] (0 or 1)
            - is_terminal: [B] (boolean mask)
    """
    batch_size = embeddings.shape[0]
    device = embeddings.device
    batch_range = torch.arange(batch_size, device=device)

    board = embeddings[..., 0].clone()  # [B, 3, 3] (+1 for P0, -1 for P1)
    current_player = embeddings[:, 0, 0, 1].long()  # [B] (0 or 1)

    # Convert flat action 0..8 to (row, col)
    row = actions_taken // 3
    col = actions_taken % 3

    piece = torch.where(current_player == 0, 1.0, -1.0)
    board[batch_range, row, col] = piece

    winner, is_terminal = check_tictactoe_winner(board)

    # Reward relative to current player
    # If game not over, reward is 0. If game over, player moving receives -outcome
    # relative to the winner in that state? No, standard zero-sum:
    # If I am player X, reward is +1 if I win, -1 if I lose, 0 if draw.
    # The winner variable is +1 for P0 win, -1 for P1 win.
    p0_win_reward = torch.where(current_player == 0, 1.0, -1.0)
    p1_win_reward = torch.where(current_player == 1, 1.0, -1.0)

    reward = torch.zeros(batch_size, device=device)
    reward = torch.where(winner == 1.0, p0_win_reward, reward)
    reward = torch.where(winner == -1.0, p1_win_reward, reward)

    next_to_play = 1 - current_player

    next_embeddings = torch.zeros_like(embeddings)
    next_embeddings[..., 0] = board
    next_embeddings[..., 1] = next_to_play.view(-1, 1, 1).expand(-1, 3, 3).float()

    return next_embeddings, reward, next_to_play, is_terminal


def get_canonical_obs(board_3x3: torch.Tensor, player: int) -> torch.Tensor:
    """
    Constructs 3-channel active-player canonical observation [1, 3, 3, 3]:
        Channel 0: Active player pieces (1.0)
        Channel 1: Opponent pieces (1.0)
        Channel 2: Active player turn plane (1.0 if player == 0 / 'X', 0.0 if player == 1 / 'O')
    """
    device = board_3x3.device
    my_piece = 1.0 if player == 0 else -1.0
    opp_piece = -1.0 if player == 0 else 1.0

    my_plane = (board_3x3 == my_piece).float()
    opp_plane = (board_3x3 == opp_piece).float()
    turn_plane = torch.full_like(my_plane, 1.0 if player == 0 else 0.0)

    canonical = torch.stack([my_plane, opp_plane, turn_plane], dim=0).unsqueeze(
        0
    )  # [1, 3, 3, 3]
    return canonical.to(device)


def embeddings_to_canonical(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Converts MCTS state embeddings [B, 3, 3, 2] into 3-channel canonical model input [B, 3, 3, 3]:
        Channel 0: Active player pieces (1.0)
        Channel 1: Opponent pieces (1.0)
        Channel 2: Active player turn plane (1.0 for Player 0 / 'X', 0.0 for Player 1 / 'O')
    """
    board = embeddings[..., 0]  # [B, 3, 3]
    player = embeddings[:, 0, 0, 1].long()  # [B]

    my_piece = torch.where(player == 0, 1.0, -1.0).view(-1, 1, 1)
    opp_piece = torch.where(player == 0, -1.0, 1.0).view(-1, 1, 1)

    my_plane = (board == my_piece).float()
    opp_plane = (board == opp_piece).float()
    turn_plane = (player == 0).float().view(-1, 1, 1).expand(-1, 3, 3)

    return torch.stack([my_plane, opp_plane, turn_plane], dim=1)  # [B, 3, 3, 3]


def get_legal_actions_mask(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Computes boolean legal action mask [B, 9] for MCTS embeddings.
    Empty cells (board == 0) are legal actions.
    """
    board = embeddings[..., 0]  # [B, 3, 3]
    flat_board = board.view(board.shape[0], -1)  # [B, 9]
    return flat_board == 0


# ============================================================================
# 4. Self-Play Episode Data Collector
# ============================================================================


def run_self_play_game(
    model: nn.Module,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, torch.Tensor]]:
    """
    Executes a single game of self-play using MCTS and active-player canonical observations.
    Returns list of training tuples (state, target_policy, target_value).
    """
    model.eval()
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

        def expansion_fn(embeddings):
            with torch.no_grad():
                canonical_x = embeddings_to_canonical(embeddings)
                logits, value = model(canonical_x)
                legal_mask = get_legal_actions_mask(embeddings)
                masked_logits = torch.where(legal_mask, logits, -1e9)
                return masked_logits, value.squeeze(-1)

        # MCTS root embedding setup [1, 3, 3, 2]
        root_embed = torch.zeros(1, 3, 3, 2, device=device)
        root_embed[0, ..., 0] = board_3x3
        root_embed[0, ..., 1] = float(player)

        # Run batched MCTS search
        tree = mcts_search(
            root_embeddings=root_embed,
            num_simulations=num_simulations,
            num_actions=9,
            expansion_fn=expansion_fn,
            dynamics_fn=tictactoe_dynamics_fn,
            root_to_play=torch.tensor([player], device=device),
            pb_c_init=C_PUCT,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            dirichlet_alpha=DIRICHLET_ALPHA,
        )

        root_visits = tree["children_visits"][0, 0]  # [9]

        # Target policy for Neural Network loss is ALWAYS regular visit count distribution (tau = 1.0)
        raw_target_policy = get_mcts_visit_policy(
            root_visits.unsqueeze(0), temperature=1.0
        ).squeeze(0)
        target_policy = torch.where(action_mask, raw_target_policy, 0.0)
        policy_sum = target_policy.sum()
        if policy_sum > 0:
            target_policy = target_policy / policy_sum
        else:
            target_policy = action_mask.float() / action_mask.float().sum()

        # Action selection temperature schedule (tau = 1.0 for first N moves, tau = 0.0 thereafter)
        temp = (
            TEMPERATURE_EXPLORATION
            if move_count <= TEMP_THRESHOLD_MOVES
            else TEMPERATURE_EXPLOITATION
        )
        action_policy = get_mcts_visit_policy(
            root_visits.unsqueeze(0), temperature=temp
        ).squeeze(0)
        action_policy = torch.where(action_mask, action_policy, 0.0)
        action_sum = action_policy.sum()
        if action_sum > 0:
            action_policy = action_policy / action_sum
        else:
            action_policy = action_mask.float() / action_mask.float().sum()

        # Sample action using functional.action_selection helpers
        if temp > 0.0:
            dist = torch.distributions.Categorical(probs=action_policy)
            action_idx_tensor, _ = sample_distribution(dist, explore=True)
            action_idx = action_idx_tensor.item()
        else:
            action_idx_tensor, _ = argmax_selector(action_policy.unsqueeze(0))
            action_idx = action_idx_tensor.squeeze().item()

        trajectory.append(
            {
                "state": canonical_obs.squeeze(0).cpu(),
                "target_policy": target_policy.cpu(),
                "player": player,
            }
        )

        # Update local board state
        row, col = action_idx // 3, action_idx % 3
        piece = 1.0 if player == 0 else -1.0
        board_3x3[row, col] = piece

        env.step(action_idx)

    # Determine final outcome z from captured PettingZoo rewards ('player_1' vs 'player_2')
    p0_reward = 0.0
    if "player_1" in final_rewards:
        p0_reward = float(final_rewards["player_1"])
    elif "player_2" in final_rewards:
        p0_reward = -float(final_rewards["player_2"])

    p1_reward = -p0_reward

    # Backfill target values z for each step relative to player at turn t
    samples = []
    for step in trajectory:
        player = step["player"]
        z = p0_reward if player == 0 else p1_reward
        samples.append(
            {
                "state": step["state"],
                "target_policy": step["target_policy"],
                "target_value": torch.tensor([z], dtype=torch.float32),
            }
        )

    return samples


# ============================================================================
# 5. Baseline Evaluator (AlphaZero vs. Random Player)
# ============================================================================


def evaluate_vs_random(
    model: nn.Module,
    num_games: int = NUM_EVAL_GAMES,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluates trained AlphaZero model against a Random agent.
    Plays half games as Player 0 ('player_1'), half as Player 1 ('player_2').
    Tracks separate P1 and P2 statistics.
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
                def expansion_fn(embeddings):
                    with torch.no_grad():
                        canonical_x = embeddings_to_canonical(embeddings)
                        logits, value = model(canonical_x)
                        legal_mask = get_legal_actions_mask(embeddings)
                        masked_logits = torch.where(legal_mask, logits, -1e9)
                        return masked_logits, value.squeeze(-1)

                root_embed = torch.zeros(1, 3, 3, 2, device=device)
                root_embed[0, ..., 0] = board_3x3
                root_embed[0, ..., 1] = float(player)

                tree = mcts_search(
                    root_embeddings=root_embed,
                    num_simulations=num_simulations,
                    num_actions=9,
                    expansion_fn=expansion_fn,
                    dynamics_fn=tictactoe_dynamics_fn,
                    root_to_play=torch.tensor([player], device=device),
                    pb_c_init=C_PUCT,
                    dirichlet_epsilon=DIRICHLET_EPSILON,
                )

                root_visits = tree["children_visits"][0, 0]
                raw_policy = get_mcts_visit_policy(
                    root_visits.unsqueeze(0), temperature=TEMPERATURE_EVAL
                ).squeeze(0)

                target_policy = torch.where(action_mask, raw_policy, 0.0)
                action_idx_tensor, _ = argmax_selector(target_policy.unsqueeze(0))
                action_idx = action_idx_tensor.squeeze().item()
            else:
                action_idx = random.choice(legal_actions)

            row, col = action_idx // 3, action_idx % 3
            piece = 1.0 if player == 0 else -1.0
            board_3x3[row, col] = piece

            env.step(action_idx)

        # Separate P1 vs P2 scoring
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
# 6. Main AlphaZero Training Loop
# ============================================================================


def train_alphazero_tictactoe():
    """
    Main AlphaZero self-play training script.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seeds for reproducibility
    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)
    random.seed(SEED)

    learner_model = TicTacToeNet(
        num_filters=NUM_FILTERS, num_res_blocks=NUM_RES_BLOCKS
    ).to(device)
    actor_model = copy.deepcopy(learner_model).to(device)

    optimizer = torch.optim.Adam(
        learner_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Initialize Replay Buffer using functional.replay_buffer
    buffer_shapes = {
        "state": (3, 3, 3),
        "target_policy": (9,),
        "target_value": (1,),
    }
    replay_buffer_state = init_buffer(
        REPLAY_BUFFER_CAPACITY, buffer_shapes, device=device
    )

    # Initialize W&B tracking
    wandb.init(
        project="alphazero-tictactoe",
        name=f"alphazero_continuous_res{NUM_RES_BLOCKS}_f{NUM_FILTERS}_sims{NUM_MCTS_SIMULATIONS}_envs{NUM_VECTOR_ENVS}",
        config={
            "total_training_steps": TOTAL_TRAINING_STEPS,
            "num_vector_envs": NUM_VECTOR_ENVS,
            "min_buffer_size": MIN_BUFFER_SIZE,
            "eval_interval": EVAL_INTERVAL,
            "param_sync_interval": PARAM_SYNC_INTERVAL,
            "num_mcts_simulations": NUM_MCTS_SIMULATIONS,
            "c_puct": C_PUCT,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "dirichlet_epsilon": DIRICHLET_EPSILON,
            "temp_threshold_moves": TEMP_THRESHOLD_MOVES,
            "batch_size": BATCH_SIZE,
            "replay_buffer_capacity": REPLAY_BUFFER_CAPACITY,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_filters": NUM_FILTERS,
            "num_res_blocks": NUM_RES_BLOCKS,
            "num_eval_games": NUM_EVAL_GAMES,
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
        # 1. Continuous Self-Play Data Collection using Actor Network
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

        # 2. Continuous 1-Step Network SGD Optimization using uniform_sample from functional.replay_buffer
        minibatch = uniform_sample(replay_buffer_state, rng_key, BATCH_SIZE)
        states = minibatch["state"]
        target_policies = minibatch["target_policy"]
        target_values = minibatch["target_value"]

        policy_logits, predicted_value = learner_model(states)

        raw_p_loss, _ = cross_entropy_loss(policy_logits, target_policies)
        policy_loss = raw_p_loss.mean()

        raw_v_loss, _ = mse_loss(predicted_value.view(-1), target_values.view(-1))
        value_loss = raw_v_loss.mean()

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. Synchronize Actor Network weights from Learner Network periodically
        if step % PARAM_SYNC_INTERVAL == 0:
            actor_model.load_state_dict(learner_model.state_dict())

        # Log continuous step metrics to W&B
        log_dict = {
            "global_step": step,
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "buffer/size": replay_buffer_state.size,
            "search/mcts_simulations": NUM_MCTS_SIMULATIONS,
            "search/c_puct": C_PUCT,
        }

        # 3. Periodic Evaluation against Random Baseline
        if step % EVAL_INTERVAL == 0 or step == TOTAL_TRAINING_STEPS:
            eval_metrics = evaluate_vs_random(
                learner_model,
                num_games=NUM_EVAL_GAMES,
                num_simulations=NUM_MCTS_SIMULATIONS,
                device=device,
            )

            log_dict.update(
                {
                    "eval/win_rate": eval_metrics["eval/win_rate"],
                    "eval/draw_rate": eval_metrics["eval/draw_rate"],
                    "eval/loss_rate": eval_metrics["eval/loss_rate"],
                    "eval/p1_win_rate": eval_metrics["eval/p1_win_rate"],
                    "eval/p1_draw_rate": eval_metrics["eval/p1_draw_rate"],
                    "eval/p1_loss_rate": eval_metrics["eval/p1_loss_rate"],
                    "eval/p2_win_rate": eval_metrics["eval/p2_win_rate"],
                    "eval/p2_draw_rate": eval_metrics["eval/p2_draw_rate"],
                    "eval/p2_loss_rate": eval_metrics["eval/p2_loss_rate"],
                }
            )

        wandb.log(log_dict)

    wandb.finish()


if __name__ == "__main__":
    train_alphazero_tictactoe()
