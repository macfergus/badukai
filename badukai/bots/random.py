import random

from .base import Bot

__all__ = [
    'RandomBot',
]


class RandomBot(Bot):
    def __init__(self, board_size):
        self._board_size = board_size

    def board_size(self):
        return self._board_size

    def select_move(self, game_state):
        return random.choice(game_state.legal_moves())

    def serialize(self, h5group):
        h5group.attrs['board_size'] = self._board_size


def load_from_hdf5(h5group):
    return RandomBot(
        board_size=h5group.attrs['board_size']
    )
