import numpy as np
from baduk import Move, Player, Point

from .base import Encoder


class LibertiesEncoder(Encoder):
    """Game state encoder with liberty and ko planes."""
    def __init__(self, board_size):
        """
        Args:
            board_size (int)
        """
        self._board_size = board_size
        self._pass_idx = self._board_size * self._board_size

        # 0 - 3. black stones with 1, 2, 3, 4+ liberties
        # 4 - 7. white stones with 1, 2, 3, 4+ liberties
        # 8. black plays next
        # 9. white plays next
        # 10. move would be illegal due to ko
        self.num_planes = 11

    def board_size(self):
        return self._board_size

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        if game_state.next_player == Player.black:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1
        for r in range(self._board_size):
            for c in range(self._board_size):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(Move.play(p)):
                        board_tensor[10][r][c] = 1
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color == Player.white:
                        liberty_plane += 4
                    board_tensor[liberty_plane][r][c] = 1

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
    return LibertiesEncoder(board_size)
