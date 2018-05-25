__all__ = [
    'BotHandler',
]

from .command import failure, success


class BotHandler:
    def __init__(self, bot):
        self.is_done = False
        self.bot = bot

    def handle_quit(self, _):
        self.is_done = True
        return success('bye!')
