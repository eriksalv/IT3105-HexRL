import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np


class Player(Enum):
    RED = 1
    BLUE = 2


class HexStateManager:
    def __init__(self, k: int) -> None:
        self.k = k
        self.winner = None
        self.board = np.zeros((self.k, self.k), dtype=np.int8)
        self.current_player = Player.RED

    def new_game(self, starting_player=Player.RED) -> None:
        """
        Hex board is represented as a square array. The RED player's goal is
        to connect the "top" and "bottom" of the array, and the BLUE player's goal
        is to connect the "left" and "right" of the array. RED player starts by default.
        The board initially contains only zeros, meaning that they are empty slots.
        Later, nodes with a "1" represent RED pieces, and "2" BLUE pieces.
        """
        self.winner = None
        self.board = np.zeros((self.k, self.k), dtype=np.int8)
        self.current_player = starting_player

    def get_legal_moves(self) -> list[tuple[int, int]]:
        coordinates = []

        for coordinate, state in np.ndenumerate(self.board):
            if state == 0:
                coordinates.append(coordinate)

        return coordinates

    def make_move(self, coordinate: tuple[int, int]):
        if self.winner is not None:
            raise ValueError('Game already over')

        if coordinate not in self.get_legal_moves():
            raise ValueError('Illegal move')

        # Place piece at coordinate
        self.board[coordinate] = self.current_player.value

        if self.is_game_over():
            self.winner = self.current_player

        self.current_player = Player.RED if self.current_player == Player.BLUE else Player.BLUE

    def get_neighbours(self, coordinate: tuple[int, int]) -> list[tuple[int, int]]:
        row, col = coordinate

        neighbours = [(row - 1, col), (row - 1, col + 1),
                      (row, col - 1), (row, col + 1),
                      (row + 1, col - 1), (row + 1, col)]

        return list(filter(lambda coord: 0 <= coord[0] < self.k and 0 <= coord[1] < self.k, neighbours))

    def get_own_pieces(self, coordinates: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return list(filter(lambda coord: self.board[coord] == self.current_player.value, coordinates))

    def is_game_over(self) -> bool:
        """
        Check if the latest move won the game for the current player.
        If current player is RED, start from the top row and search towards
        the bottom. If BLUE, go from left to right column.
        """
        starting_coordinates = [(0, col) for col in range(self.k)] if self.current_player == Player.RED \
            else [(row, 0) for row in range(self.k)]

        starting_pieces = self.get_own_pieces(starting_coordinates)

        visited = {}

        for piece in starting_pieces:
            if self.traverse(piece, visited) is True:
                return True

        return False

    def is_final(self):
        return self.winner is not None

    def traverse(self, piece, visited):
        """
        Traverses current players pieces DEPTH FIRST (recursively). Only edges between
        pieces of the current player are considered valid edges for traversal
        """
        # Check if goal is reached
        if (self.current_player == Player.RED and piece[0] == self.k - 1) or \
                (self.current_player == Player.BLUE and piece[1] == self.k - 1):
            return True

        visited[piece] = True
        for neighbour_piece in self.get_own_pieces(self.get_neighbours(piece)):
            if neighbour_piece not in visited:
                if self.traverse(neighbour_piece, visited):
                    return True
        return False

    def get_state(self) -> tuple[np.ndarray, int]:
        return self.board.copy(), self.current_player.value

    def get_possible_states(self, legal_moves: list[tuple[int, int]]) -> list[tuple[np.ndarray, int]]:
        possible_states = []

        for move in legal_moves:
            board = self.board.copy()
            board[move] = self.current_player.value
            next_player = Player.RED.value if self.current_player == Player.BLUE else Player.BLUE.value
            possible_states.append((board, next_player))

        return possible_states

    def make_random_starting_move(self):
        middle_line = [(i, j) for i in range(self.k) for j in range(self.k) if i + j == self.k - 1]
        move = random.choice(middle_line)
        self.make_move(move)

    def show_board(self, block=True):
        """
        There is probably an easier way to visualize the board, but this was too much fun
        """
        enumerated = list(np.ndenumerate(self.board))
        coords = np.array([enum[0] for enum in enumerated])

        horizontal_grids = coords.reshape((self.k, self.k, 2))
        vertical_grids = np.flip(horizontal_grids)

        diag_offset = self.k - 2
        diagonal_grids = [np.flipud(horizontal_grids).diagonal(i).T
                          for i in range(-diag_offset, diag_offset + 1)]

        theta = - np.pi / 4
        rot_neg45 = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])

        flip = np.array([[-1, 0],
                         [0, -1]])

        rot_coords = coords.dot(flip).dot(rot_neg45)

        xs = rot_coords[:, 0]
        ys = rot_coords[:, 1]

        col = np.where(self.board == 1, 'red',
                       np.where(self.board == 2, 'blue',
                                'black')).flatten()

        plt.figure(figsize=(10, 15))
        plt.axis('off')
        plt.scatter(xs, ys, s=200, c=col)

        if self.winner == Player.RED:
            plt.title('Red wins', c='red')
        elif self.winner == Player.BLUE:
            plt.title('Blue wins', c='blue')

        for i, coord in enumerate(enumerated):
            plt.annotate(coord[0], (xs[i], ys[i]), (xs[i] + 0.1, ys[i] - 0.1))

        for lines in [horizontal_grids, vertical_grids]:
            lines = lines.dot(flip).dot(rot_neg45)
            xs = lines[:, :, 0]
            ys = lines[:, :, 1]
            plt.plot(xs, ys, c='black', zorder=-1)

        for line in diagonal_grids:
            line = line.dot(flip).dot(rot_neg45)
            xs = line[:, 0]
            ys = line[:, 1]
            plt.plot(xs, ys, c='black', zorder=-1)

        plt.show(block=block)


def mark_bridge_endpoints(board):
    k = len(board)
    # We only need to check in the "downward" direction because of symmetry
    possible_endpoint_offsets = [(1, 1), (2, -1), (-1, 2)]
    possible_bridge_carrier_offsets = [((1, 0), (0, 1)), ((1, -1), (1, 0)), ((-1, 1), (0, 1))]

    for row in range(k):
        for col in range(k):
            # Check if the current cell contains a stone
            if board[row][col] in [1, 2, 3, 4]:
                for (dr, dc), (carrier1, carrier2) in zip(possible_endpoint_offsets, possible_bridge_carrier_offsets):
                    r, c = row + dr, col + dc
                    # Check if the diagonal cell is within the board boundaries and that endpoints have the same color
                    if 0 <= r < k and 0 <= c < k and board[r, c] != 0 and board[r][c] % 2 == board[row][col] % 2:
                        # Check if both possible bridge carriers are empty
                        if board[row + carrier1[0], col + carrier1[1]] == 0 and \
                                board[row + carrier2[0], col + carrier2[1]] == 0:
                            # Mark the bridge endpoints with 3 for player 1 and 4 for player 2
                            if board[row][col] % 2 == 1:
                                board[row, col] = 3
                                board[r, c] = 3
                            else:
                                board[row, col] = 4
                                board[r, c] = 4

    return board


if __name__ == "__main__":
    state_manager = HexStateManager(4)
    state_manager.new_game()
    state_manager.make_move((1, 1))
    state_manager.make_move((3, 0))
    state_manager.make_move((0, 0))
    state_manager.make_move((2, 2))
    state_manager.make_move((0, 2))
    state_manager.make_move((0, 3))

    print(state_manager.board)

    state_manager.show_board(block=False)
    new_board = mark_bridge_endpoints(state_manager.board)
    print(new_board)
    plt.show()
