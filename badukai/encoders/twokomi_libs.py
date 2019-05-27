import numpy as np
from baduk import Move, Player, Point

from .base import Encoder


class TwoKomiLibertyEncoder(Encoder):
    """Game state encoder with liberty and ko planes."""
    def __init__(self, board_size):
        """
        Args:
            board_size (int)
        """
        self._board_size = board_size
        self._pass_idx = self._board_size * self._board_size

        # 0. our player stones with 1 liberty
        # 1. our stones with 2 liberties
        # 2. our stones with 3+ liberties
        # 3. opponent stones with 1 liberty
        # 4. opponent stones with 2 liberty
        # 5. opponent stones with 3+ liberty
        # 6. move is illegal due to ko
        # 7. 1 if we get komi
        # 8. 1 if opponent gets komi
        self.num_planes = 9

        self._points = []
        for r in range(self._board_size):
            for c in range(self._board_size):
                self._points.append(Point(row=r + 1, col=c + 1))

    def board_size(self):
        return self._board_size

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        # Positive komi is for white, negative is for black.
        komi = game_state.komi()
        if game_state.next_player == Player.black:
            komi *= -1
            black_plane = 0
            white_plane = 3
        else:
            black_plane = 3
            white_plane = 0

        # Two separate komi planes, can handle handicap or reverse komi
        # (but not big komis)
        if komi > 4:
            board_tensor[7] = 1
        elif komi < -4:
            board_tensor[8] = 1
        board_tensor[6] = game_state.ko_points_as_array()

        board = game_state.board
        board_tensor[black_plane] = \
            board.stones_with_n_liberties_as_array(Player.black, 1)
        board_tensor[black_plane + 1] = \
            board.stones_with_n_liberties_as_array(Player.black, 2)
        board_tensor[black_plane + 2] = \
            board.stones_with_min_liberties_as_array(Player.black, 3)
        board_tensor[white_plane] = \
            board.stones_with_n_liberties_as_array(Player.white, 1)
        board_tensor[white_plane + 1] = \
            board.stones_with_n_liberties_as_array(Player.white, 2)
        board_tensor[white_plane + 2] = \
            board.stones_with_min_liberties_as_array(Player.white, 3)

        return board_tensor

    def encode_move(self, move):
        """Turn a board point into an integer index."""
        if move.is_play:
            # Points are 1-indexed
            return self._board_size * (move.point.row - 1) + \
                (move.point.col - 1)
        elif move.is_pass:
            return self._pass_idx
        raise ValueError('Cannot encode resign move')

    def decode_move_index(self, index):
        """Turn an integer index into a board point."""
        if index == self._pass_idx:
            return Move.pass_turn()
        row = index // self._board_size
        col = index % self._board_size
        return Move.play(Point(row=row + 1, col=col + 1))

    def num_moves(self):
        # Add 1 for pass
        return self._board_size * self._board_size + 1

    def shape(self):
        return (self.num_planes, self._board_size, self._board_size)


def create(board_size):
    return TwoKomiLibertyEncoder(board_size)
