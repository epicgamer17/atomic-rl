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

TODO: training on a similarly sized model takes way longer than it did before. figure out why and how the current system can be improved and optimized to run faster. a training run used to take about 20 minutes for 20k steps on MuZero with 3 Resnet blocks per component and 24 filters each, batch size 8, 25 simulations, 3 torch mp actor processes, and batched mcts with a search batch size of 5, also unroll of 5. now its well over an hour, probably over 2 with the alphazero params which should be faster in terms of wall clock time. on mac on cpu
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


from envs.functions.tictactoe import (
    check_tictactoe_winner,
    tictactoe_dynamics_fn,
    get_canonical_obs,
    embeddings_to_canonical,
    get_legal_actions_mask,
)


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
    Uses fast tensor board dynamics directly, eliminating environment dictionary and string parsing overhead.
    """
    model.eval()
    board = torch.zeros((3, 3), device=device)
    player = 0
    move_count = 0
    trajectory = []

    def expansion_fn(embeddings):
        with torch.no_grad():
            canonical_x = embeddings_to_canonical(embeddings)
            logits, value = model(canonical_x)
            return logits, value.squeeze(-1)

    while True:
        move_count += 1
        action_mask = (board.view(-1) == 0)

        canonical_obs = get_canonical_obs(board, player)

        root_embed = board.new_zeros(1, 3, 3, 2)
        root_embed[0, ..., 0] = board
        root_embed[0, ..., 1] = float(player)

        # Run batched MCTS search
        tree = mcts_search(
            root_embeddings=root_embed,
            num_simulations=num_simulations,
            num_actions=9,
            expansion_fn=expansion_fn,
            dynamics_fn=tictactoe_dynamics_fn,
            root_to_play=board.new_tensor([player], dtype=torch.long),
            root_legal_mask=action_mask.unsqueeze(0),
            pb_c_init=C_PUCT,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            dirichlet_alpha=DIRICHLET_ALPHA,
        )

        root_visits = tree["children_visits"][0, 0]  # [9]

        # Target policy for Neural Network loss is ALWAYS regular visit count distribution (tau = 1.0)
        target_policy = get_mcts_visit_policy(
            root_visits.unsqueeze(0), temperature=1.0
        ).squeeze(0)

        # Action selection temperature schedule (tau = 1.0 for first N moves, tau = 0.0 thereafter)
        temp = (
            TEMPERATURE_EXPLORATION
            if move_count <= TEMP_THRESHOLD_MOVES
            else TEMPERATURE_EXPLOITATION
        )
        action_policy = get_mcts_visit_policy(
            root_visits.unsqueeze(0), temperature=temp
        ).squeeze(0)

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

        # Update local board state directly
        row, col = action_idx // 3, action_idx % 3
        piece = 1.0 if player == 0 else -1.0
        board[row, col] = piece

        winner, is_terminal = check_tictactoe_winner(board.unsqueeze(0))

        if is_terminal.item():
            p0_reward = winner.item()
            p1_reward = -p0_reward
            samples = []
            for step in trajectory:
                pl = step["player"]
                z = p0_reward if pl == 0 else p1_reward
                samples.append(
                    {
                        "state": step["state"],
                        "target_policy": step["target_policy"],
                        "target_value": torch.tensor([z], dtype=torch.float32),
                    }
                )
            return samples

        player = 1 - player


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
                        return logits, value.squeeze(-1)

                root_embed = board_3x3.new_zeros(1, 3, 3, 2)
                root_embed[0, ..., 0] = board_3x3
                root_embed[0, ..., 1] = float(player)

                tree = mcts_search(
                    root_embeddings=root_embed,
                    num_simulations=num_simulations,
                    num_actions=9,
                    expansion_fn=expansion_fn,
                    dynamics_fn=tictactoe_dynamics_fn,
                    root_to_play=board_3x3.new_tensor([player], dtype=torch.long),
                    root_legal_mask=action_mask.unsqueeze(0),
                    pb_c_init=C_PUCT,
                    dirichlet_epsilon=0.0,
                )

                root_visits = tree["children_visits"][0, 0]
                action_policy = get_mcts_visit_policy(
                    root_visits.unsqueeze(0), temperature=TEMPERATURE_EVAL
                ).squeeze(0)
                action_idx_tensor, _ = argmax_selector(action_policy.unsqueeze(0))
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
