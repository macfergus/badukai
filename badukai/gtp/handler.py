__all__ = [
    'BotHandler',
]

from .command import failure, success


class BotHandler:
    def __init__(self, bot):
        self.is_done = False
        self.bot = bot
        self.board_size = self.bot.board_size()

    def handle_quit(self):
        self.is_done = True
        return success('bye!')

    def handle_boardsize(self, board_size):
        board_size = int(board_size)
        if board_size != self.board_size:
            return failure('only support {}x{}'.format(
                self.board_size, self.board_size))
        return success('{}'.format(board_size))
