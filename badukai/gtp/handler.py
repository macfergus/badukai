__all__ = [
    'BotHandler',
]

import baduk

from .command import failure, success

COLS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'


def parse_gtp_coords(gtp_coords):
    col = COLS.index(gtp_coords[0]) + 1
    row = int(gtp_coords[1:])
    return baduk.Point(row, col)


def parse_gtp_color(color):
    if color.lower() == 'b':
        return baduk.Player.black
    if color.lower() == 'w':
        return baduk.Player.white
    raise ValueError(color)


def encode_gtp_move(move):
    if move.is_resign:
        return 'resign'
    if move.is_pass:
        return 'pass'
    col_idx = move.point.col - 1
    return '{}{}'.format(COLS[col_idx], move.point.row)


class BotHandler:
    def __init__(self, bot):
        self.is_done = False
        self.bot = bot
        self.board_size = self.bot.board_size()
        self.board = baduk.Board(self.board_size, self.board_size)
        self.komi = 7.5
        self.game = baduk.GameState.from_board(
            self.board, baduk.Player.black, self.komi)

    def handle_quit(self):
        self.is_done = True
        return success('bye!')

    def handle_name(self):
        return success('hi')

    def handle_version(self):
        return success('1')

    def handle_protocol_version(self):
        return success('2')

    def handle_list_commands(self):
        return success('some')

    def handle_komi(self, komi):
        self.komi = float(komi)
        self.game = baduk.GameState.from_board(
            self.board, baduk.Player.black, self.komi)
        return success('ok')

    def handle_boardsize(self, board_size):
        board_size = int(board_size)
        if board_size != self.board_size:
            return failure('only support {}x{}'.format(
                self.board_size, self.board_size))
        return success('{}'.format(board_size))

    def handle_clear_board(self):
        self.board = baduk.Board(self.board_size, self.board_size)
        self.game = baduk.GameState.from_board(
            self.board, baduk.Player.black, self.komi)
        return success('cleared')

    def handle_play(self, color, coords):
        player = parse_gtp_color(color)
        point = parse_gtp_coords(coords)
        if player != self.game.next_player:
            return failure('wrong player')
        self.game = self.game.apply_move(baduk.Move(point))
        return success('ok')

    def handle_genmove(self, color):
        player = parse_gtp_color(color)
        if player != self.game.next_player:
            return failure('wrong player')
        move = self.bot.select_move(self.game)
        self.game = self.game.apply_move(move)
        return success(encode_gtp_move(move))
