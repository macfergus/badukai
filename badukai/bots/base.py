__all__ = [
    'Bot',
]


class Bot(object):
    def name(self):
        return 'Bot'

    def board_size(self):
        """Return the size of game board this bot supports.

        Returns:
            int
        """
        raise NotImplementedError()

    def select_move(self, game_state):
        raise NotImplementedError()

    def serialize(self, h5group):
        """Serialize the bot to an HDF5 group."""

    def set_option(self, name, value):
        """Set an engine-specific option"""
