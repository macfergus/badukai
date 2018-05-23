import sys

__all__ = [
    'BotRunner',
]


class BotRunner:
    def __init__(self, bot, inf=None, outf=None):
        self.bot = bot

        if inf:
            self.inf = inf
        else:
            self.inf = sys.stdin

        if outf:
            self.outf = outf
        else:
            self.outf = sys.stdin

    def run_forever(self):
        pass
