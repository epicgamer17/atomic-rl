from typing import Tuple
import torch


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

    winner = board.new_zeros(batch_size)
    winner = torch.where(p0_wins, board.new_tensor(1.0), winner)
    winner = torch.where(p1_wins, board.new_tensor(-1.0), winner)

    is_terminal = p0_wins | p1_wins | board_full
    return winner, is_terminal


def tictactoe_dynamics_fn(
    embeddings: torch.Tensor, actions_taken: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transition dynamics for MCTS simulation.

    Args:
        embeddings: State tensor [B, 3, 3, 2] (board, to_play)
        actions_taken: Action indices [B] in 0..8.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - next_embeddings: [B, 3, 3, 2]
            - reward: [B] (terminal reward relative to current player)
            - next_to_play: [B] (0 or 1)
            - is_terminal: [B] (boolean mask)
            - next_legal_mask: [B, 9] (boolean mask)
    """
    batch_size = embeddings.shape[0]
    batch_range = torch.arange(batch_size, device=embeddings.device)

    board = embeddings[..., 0].clone()  # [B, 3, 3] (+1 for P0, -1 for P1)
    current_player = embeddings[:, 0, 0, 1].long()  # [B] (0 or 1)

    # Convert flat action 0..8 to (row, col)
    row = actions_taken // 3
    col = actions_taken % 3

    piece = torch.where(current_player == 0, 1.0, -1.0)
    board[batch_range, row, col] = piece

    winner, is_terminal = check_tictactoe_winner(board)

    p0_win_reward = torch.where(current_player == 0, 1.0, -1.0)
    p1_win_reward = torch.where(current_player == 1, 1.0, -1.0)

    reward = embeddings.new_zeros(batch_size)
    reward = torch.where(winner == 1.0, p0_win_reward, reward)
    reward = torch.where(winner == -1.0, p1_win_reward, reward)

    next_to_play = 1 - current_player

    next_embeddings = torch.zeros_like(embeddings)
    next_embeddings[..., 0] = board
    next_embeddings[..., 1] = next_to_play.view(-1, 1, 1).expand(-1, 3, 3).float()

    next_legal_mask = (board.view(batch_size, -1) == 0)

    return next_embeddings, reward, next_to_play, is_terminal, next_legal_mask


def get_canonical_obs(board_3x3: torch.Tensor, player: int) -> torch.Tensor:
    """
    Constructs 3-channel active-player canonical observation [1, 3, 3, 3]:
        Channel 0: Active player pieces (1.0)
        Channel 1: Opponent pieces (1.0)
        Channel 2: Active player turn plane (1.0 if player == 0 / 'X', 0.0 if player == 1 / 'O')
    """
    my_piece = 1.0 if player == 0 else -1.0
    opp_piece = -1.0 if player == 0 else 1.0

    my_plane = (board_3x3 == my_piece).float()
    opp_plane = (board_3x3 == opp_piece).float()
    turn_plane = board_3x3.new_full(my_plane.shape, 1.0 if player == 0 else 0.0)

    return torch.stack([my_plane, opp_plane, turn_plane], dim=0).unsqueeze(0)


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
