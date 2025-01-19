# We will define a othello game represented by a 8x8 matrix
# we need to implement the following functions:
# 1. init_board(): initialize the board
# 2. print_board(): print the board
# 3. legal_moves(): return a list of legal moves
# 4. make_move(): make a move and flip the pieces
# 5. game_over(): check if the game is over

import numpy as np
import random
from tqdm import tqdm
import os
import argparse

def init_board():
    board = np.zeros((8, 8))
    board[3, 3] = 1
    board[3, 4] = -1
    board[4, 3] = -1
    board[4, 4] = 1
    return board


def print_board(board):
    print("  0 1 2 3 4 5 6 7")
    for i in range(8):
        print(i, end=" ")
        for j in range(8):
            if board[i, j] == 1:
                print("@", end=" ")
            elif board[i, j] == -1:
                print("O", end=" ")
            else:
                print(".", end=" ")
        print()


def legal_moves(board, player):
    legal_moves = []
    opponent = -player
    for r in range(8):
        for c in range(8):
            if board[r, c] == 0:  # Check if the cell is empty
                # Check all 8 directions
                for dr, dc in [
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == opponent:
                        # Found an opponent's piece in the direction, now keep going
                        while 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == opponent:
                            nr += dr
                            nc += dc
                        if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == player:
                            # Found a player's piece at the end of the line, so it's a legal move
                            legal_moves.append((r, c))
                            break  # No need to check other directions for this cell
    return legal_moves


def make_move(board, player, move):
    if move == (-1, -1):  # Handle skip move
        return board  # Board remains unchanged
    r, c = move
    if board[r, c] != 0:
        raise ValueError("Invalid move: Cell is not empty")

    opponent = -player
    board[r, c] = player  # Place the player's piece

    # Check all 8 directions for pieces to flip
    for dr, dc in [
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]:
        row, col = r + dr, c + dc
        pieces_to_flip = []
        while 0 <= row < 8 and 0 <= col < 8 and board[row, col] == opponent:
            pieces_to_flip.append((row, col))
            row += dr
            col += dc
        if 0 <= row < 8 and 0 <= col < 8 and board[row, col] == player:
            # Flip the pieces in this direction
            for flip_r, flip_c in pieces_to_flip:
                board[flip_r, flip_c] = player
    return board


def game_over(board):
    # Game is over if there are no legal moves for either player
    if not legal_moves(board, 1) and not legal_moves(board, -1):
        return True
    # Game is also over if the board is full
    if np.all(board != 0):
        return True
    return False


def play_random_game(silent=True, return_final_board=False):
    board = init_board()
    player = -1
    turn = 0
    moves_played = []
    while not game_over(board):
        if not silent:
            print(f"Turn {turn}, Player {'@' if player == 1 else 'O'}'s turn:")
            print_board(board)
        possible_moves = legal_moves(board, player)
        if possible_moves:
            move = random.choice(possible_moves)
            if not silent:
                print(
                    f"Player {'@' if player == 1 else 'O'} plays at {move[0]+1},{move[1]+1}"
                )
            board = make_move(board, player, move)
            moves_played.append(move)
            player *= -1  # Switch player
        else:
            if not silent:
                print(
                    f"No legal moves for Player {'@' if player == 1 else 'O'}. Skipping turn."
                )
            move = (-1, -1)  # Indicate that the player skipped the turn
            moves_played.append(move)
            player *= -1  # Switch player, hoping the other player has moves
            if not silent and not legal_moves(
                board, player
            ):  # If the other player also has no moves, game is over
                print("No legal moves for either player. Game over.")
                break
        turn += 1

    score_x = np.sum(board == 1)
    score_o = np.sum(board == -1)
    if not silent:
        print("Game Over!")
        print_board(board)
        print(f"Final Score: @ = {score_x}, O = {score_o}")
        if score_x > score_o:
            print("Winner: @")
        elif score_o > score_x:
            print("Winner: O")
        else:
            print("It's a draw!")
    if return_final_board:
        return moves_played, board
    return moves_played


def apply_moves(board, moves, initial_player):
    player = initial_player
    current_board = np.copy(
        board
    )  # Work on a copy to avoid modifying the original board

    for move in moves:
        if not isinstance(move, tuple) or len(move) != 2:
            raise ValueError(
                f"Invalid move format: {move}. Moves should be tuples (row, col)."
            )
        r, c = move

        moves = legal_moves(current_board, player)

        if not moves:
            if (r, c) != (-1, -1):
                raise ValueError(
                    f"No legal moves for Player {'@' if player == 1 else 'O'} but {move} was played."
                )
        elif (r, c) == (-1, -1):
            raise ValueError(
                f"Player {'@' if player == 1 else 'O'} should have played at {random.choice(moves)} but skipped turn."
            )
        elif (r, c) not in moves:
            raise ValueError(f"Legal moves are: {moves} but {move} was played.")

        current_board = make_move(current_board, player, (r, c))
        player *= -1  # Switch player

    return current_board


def _play_random_game():
    board = init_board()
    player = 1
    moves_played = []
    while not game_over(board):
        possible_moves = legal_moves(board, player)
        if possible_moves:
            move = random.choice(possible_moves)
            board = make_move(board, player, move)
            moves_played.append(move)
            player *= -1
        else:
            moves_played.append((-1, -1))  # Record skip move
            player *= -1
            if not legal_moves(board, player):
                break
    return moves_played


def generate_move_dataset(n_moves, save_dir, seed):
    random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(n_moves)):
        moves_played = _play_random_game()
        moves_played = np.array([x + 8*y for (x, y) in moves_played], dtype=np.int8)
        np.save(f"{save_dir}/game_{seed}_{i}.npy", moves_played)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("--dir", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(args)
    generate_move_dataset(args.n, args.dir, args.seed)