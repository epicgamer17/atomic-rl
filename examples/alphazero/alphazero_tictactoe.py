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
from pettingzoo.classic import tictactoe_v3


# ---------------------------------------------------------------------------
# Hyperparameters & Constants (Matching AlphaZero Paper Conventions)
# ---------------------------------------------------------------------------
# Self-Play & MCTS Simulation Parameters
TOTAL_TRAINING_STEPS = (
    1000  # Total continuous training steps (1 SGD step per training loop)
)
GAMES_PER_STEP = 1  # Self-play games generated per training step
MIN_BUFFER_SIZE = 64  # Warmup buffer size before SGD optimization begins
EVAL_INTERVAL = 50  # Evaluate vs Random agent every N training steps
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
BATCH_SIZE = 64
REPLAY_BUFFER_CAPACITY = 5000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization weight decay c = 10^-4
NUM_FILTERS = 16  # 16 filters per ResNet block
NUM_RES_BLOCKS = 6

# Evaluation & Seed
NUM_EVAL_GAMES = 50
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
# 2. Inlined AlphaZero Composite Loss Function
# ============================================================================


def alphazero_loss(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    predicted_value: torch.Tensor,
    target_value: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    AlphaZero composite loss function.

    Formula:
        L = MSE(predicted_value, target_value) - sum(target_policy * log_softmax(policy_logits))

    Args:
        policy_logits: Predicted action logits [B, 9].
        target_policy: MCTS visit target policy distribution [B, 9].
        predicted_value: Predicted value scalar [B, 1] or [B].
        target_value: Game outcome z [B, 1] or [B] in [-1, +1].

    Returns:
        Tuple[torch.Tensor, Dict]: (total_loss, metrics_dict)
    """
    # Policy Cross-Entropy: - sum(pi * log(p))
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

    # Value MSE: (v - z)^2
    value_loss = F.mse_loss(predicted_value.view(-1), target_value.view(-1))

    total_loss = policy_loss + value_loss

    info = {
        "loss/total": total_loss.detach(),
        "loss/policy": policy_loss.detach(),
        "loss/value": value_loss.detach(),
    }
    return total_loss, info


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

        # Sample action for self-play environment execution
        if temp > 0.0:
            action_idx = torch.multinomial(action_policy, num_samples=1).item()
        else:
            action_idx = action_policy.argmax().item()

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


def render_tictactoe_board(board_3x3: torch.Tensor) -> str:
    """
    Renders 3x3 board tensor into ASCII representation.
        +1 -> 'X' (Player 1 / P0)
        -1 -> 'O' (Player 2 / P1)
         0 -> '.' (Empty)
    """
    symbols = {1.0: "X", -1.0: "O", 0.0: "."}
    rows = []
    for r in range(3):
        row_str = " | ".join(symbols[board_3x3[r, c].item()] for c in range(3))
        rows.append("  " + row_str)
    return "\n  ---+---+---\n".join(rows)


# ============================================================================
# 5. Baseline Evaluator (AlphaZero vs. Random Player)
# ============================================================================


def evaluate_vs_random(
    model: nn.Module,
    num_games: int = NUM_EVAL_GAMES,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    device: torch.device = torch.device("cpu"),
    render_lost_games: bool = True,
) -> Dict[str, float]:
    """
    Evaluates trained AlphaZero model against a Random agent.
    Plays half games as Player 0 ('player_1'), half as Player 1 ('player_2').
    Tracks separate P1 and P2 statistics, and ONLY prints step-by-step logs for lost games.
    """
    model.eval()
    p1_wins, p1_draws, p1_losses = 0, 0, 0
    p2_wins, p2_draws, p2_losses = 0, 0, 0
    lost_games_printed = 0

    for game_i in range(num_games):
        az_player = 0 if game_i % 2 == 0 else 1
        az_agent_name = "player_1" if az_player == 0 else "player_2"
        env = tictactoe_v3.env()
        env.reset()

        board_3x3 = torch.zeros(3, 3, device=device)
        az_reward = 0.0
        step_idx = 0
        game_history = []
        piece_sym = "X" if az_player == 0 else "O"

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if agent == az_agent_name and reward != 0:
                az_reward = reward

            if termination or truncation:
                env.step(None)
                continue

            step_idx += 1
            player = 0 if agent == "player_1" else 1
            action_mask = torch.tensor(
                obs["action_mask"], device=device, dtype=torch.bool
            )
            legal_actions = action_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

            if player == az_player:
                canonical_obs = get_canonical_obs(board_3x3, player)
                with torch.no_grad():
                    init_logits, init_val_tensor = model(canonical_obs)
                    masked_init_logits = torch.where(
                        action_mask.unsqueeze(0), init_logits, -1e9
                    )
                    initial_prior_p = F.softmax(masked_init_logits, dim=-1).squeeze(0)
                    pred_v = init_val_tensor.squeeze().item()

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
                    dirichlet_epsilon=0.0,
                )

                root_visits = tree["children_visits"][0, 0]
                raw_policy = get_mcts_visit_policy(
                    root_visits.unsqueeze(0), temperature=TEMPERATURE_EVAL
                ).squeeze(0)

                target_policy = torch.where(action_mask, raw_policy, 0.0)
                action_idx = target_policy.argmax().item()

                game_history.append(
                    {
                        "step": step_idx,
                        "player_name": f"AlphaZero ('{piece_sym}')",
                        "board": board_3x3.clone(),
                        "pred_v": pred_v,
                        "initial_prior": initial_prior_p.clone(),
                        "raw_policy": raw_policy.clone(),
                        "action_idx": action_idx,
                    }
                )
            else:
                action_idx = random.choice(legal_actions)
                opp_sym = "O" if az_player == 0 else "X"
                game_history.append(
                    {
                        "step": step_idx,
                        "player_name": f"Random ('{opp_sym}')",
                        "board": board_3x3.clone(),
                        "action_idx": action_idx,
                    }
                )

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

        # Post-Mortem analysis ONLY if AlphaZero lost this game
        if render_lost_games and (az_reward < 0) and (lost_games_printed < 2):
            lost_games_printed += 1
            print(
                f"\n!!! [LOST GAME POST-MORTEM #{lost_games_printed} (Game {game_i}, AlphaZero as {'P1' if az_player==0 else 'P2'} ('{piece_sym}'))] !!!",
                flush=True,
            )
            for h in game_history:
                if "pred_v" in h:
                    p_str = ", ".join(f"{p:.2f}" for p in h["initial_prior"].tolist())
                    pi_str = ", ".join(f"{p:.2f}" for p in h["raw_policy"].tolist())
                    print(f"\nMove Step {h['step']} | {h['player_name']}:", flush=True)
                    print(render_tictactoe_board(h["board"]), flush=True)
                    print(
                        f"  Predicted Value (v_theta) : {h['pred_v']:+.3f}", flush=True
                    )
                    print(f"  Initial Model Prior (p)   : [{p_str}]", flush=True)
                    print(f"  Final MCTS Search (pi)   : [{pi_str}]", flush=True)
                    print(
                        f"  Selected Action Index    : {h['action_idx']} (row {h['action_idx']//3}, col {h['action_idx']%3})",
                        flush=True,
                    )
                else:
                    print(
                        f"\nMove Step {h['step']} | {h['player_name']} Action: {h['action_idx']} (row {h['action_idx']//3}, col {h['action_idx']%3})",
                        flush=True,
                    )
            print(f"\nFinal Lost Board State:")
            print(render_tictactoe_board(board_3x3))
            print("=" * 60 + "\n", flush=True)

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
    print(f"[AlphaZero TicTacToe] Starting training on device: {device}", flush=True)

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)

    model = TicTacToeNet(num_filters=NUM_FILTERS, num_res_blocks=NUM_RES_BLOCKS).to(
        device
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Initialize Replay Buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_CAPACITY)

    # Initialize W&B tracking
    wandb.init(
        project="alphazero-tictactoe",
        name=f"alphazero_continuous_res{NUM_RES_BLOCKS}_f{NUM_FILTERS}_sims{NUM_MCTS_SIMULATIONS}",
        config={
            "total_training_steps": TOTAL_TRAINING_STEPS,
            "games_per_step": GAMES_PER_STEP,
            "min_buffer_size": MIN_BUFFER_SIZE,
            "eval_interval": EVAL_INTERVAL,
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

    print("\n--- Initial Evaluation (Random Weights) ---", flush=True)
    initial_eval = evaluate_vs_random(
        model,
        num_games=NUM_EVAL_GAMES,
        num_simulations=NUM_MCTS_SIMULATIONS,
        device=device,
    )
    print(
        f"Initial vs Random -> Overall Win: {initial_eval['eval/win_rate']*100:.0f}% | "
        f"P1 (Win: {initial_eval['eval/p1_win_rate']*100:.0f}%, Draw: {initial_eval['eval/p1_draw_rate']*100:.0f}%, Loss: {initial_eval['eval/p1_loss_rate']*100:.0f}%) | "
        f"P2 (Win: {initial_eval['eval/p2_win_rate']*100:.0f}%, Draw: {initial_eval['eval/p2_draw_rate']*100:.0f}%, Loss: {initial_eval['eval/p2_loss_rate']*100:.0f}%)",
        flush=True,
    )
    print("------------------------------------------\n", flush=True)

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
        # 1. Continuous Self-Play Data Collection (Paper-Faithful Asynchronous Flow)
        new_samples = []
        for _ in range(GAMES_PER_STEP):
            game_samples = run_self_play_game(
                model, num_simulations=NUM_MCTS_SIMULATIONS, device=device
            )
            new_samples.extend(game_samples)

        replay_buffer.extend(new_samples)

        if len(replay_buffer) < MIN_BUFFER_SIZE:
            continue

        # 2. Continuous 1-Step Network SGD Optimization (Paper-Faithful)
        minibatch = random.sample(replay_buffer, BATCH_SIZE)
        states = torch.stack([s["state"] for s in minibatch]).to(device)
        target_policies = torch.stack([s["target_policy"] for s in minibatch]).to(
            device
        )
        target_values = torch.stack([s["target_value"] for s in minibatch]).to(device)

        policy_logits, predicted_value = model(states)

        loss, info = alphazero_loss(
            policy_logits=policy_logits,
            target_policy=target_policies,
            predicted_value=predicted_value,
            target_value=target_values,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log continuous step metrics to W&B
        log_dict = {
            "global_step": step,
            "loss/total": info["loss/total"].item(),
            "loss/policy": info["loss/policy"].item(),
            "loss/value": info["loss/value"].item(),
            "buffer/size": len(replay_buffer),
            "search/mcts_simulations": NUM_MCTS_SIMULATIONS,
            "search/c_puct": C_PUCT,
        }

        # 3. Periodic Evaluation against Random Baseline
        if step % EVAL_INTERVAL == 0 or step == TOTAL_TRAINING_STEPS:
            eval_metrics = evaluate_vs_random(
                model,
                num_games=NUM_EVAL_GAMES,
                num_simulations=NUM_MCTS_SIMULATIONS,
                device=device,
            )

            print(
                f"Step [{step:04d}/{TOTAL_TRAINING_STEPS}] | "
                f"Loss: {info['loss/total'].item():.4f} (P: {info['loss/policy'].item():.4f}, V: {info['loss/value'].item():.4f}) | "
                f"Overall Win: {eval_metrics['eval/win_rate']*100:.0f}% | "
                f"P1 Win: {eval_metrics['eval/p1_win_rate']*100:.0f}% (L: {eval_metrics['eval/p1_loss_rate']*100:.0f}%) | "
                f"P2 Win: {eval_metrics['eval/p2_win_rate']*100:.0f}% (L: {eval_metrics['eval/p2_loss_rate']*100:.0f}%)",
                flush=True,
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
    print("\n[AlphaZero TicTacToe] Training Complete!", flush=True)


if __name__ == "__main__":
    train_alphazero_tictactoe()
