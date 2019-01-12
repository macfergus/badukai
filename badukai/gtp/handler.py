__all__ = [
    'BotHandler',
]

import baduk

from .command import failure, success

COLS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'

HANDICAP_STONES = {
    2: ['D4', 'Q16'],
    3: ['D4', 'Q16', 'D16'],
    4: ['D4', 'Q16', 'D16', 'Q4'],
    5: ['D4', 'Q16', 'D16', 'Q4', 'K10'],
    6: ['D4', 'Q16', 'D16', 'Q4', 'D10', 'Q10'],
    7: ['D4', 'Q16', 'D16', 'Q4', 'D10', 'Q10', 'K10'],
    8: ['D4', 'Q16', 'D16', 'Q4', 'D10', 'Q10', 'K4', 'K16'],
    9: ['D4', 'Q16', 'D16', 'Q4', 'D10', 'Q10', 'K4', 'K16', 'K10'],
}


def parse_gtp_coords(gtp_coords):
    col = COLS.index(gtp_coords.upper()[0]) + 1
    row = int(gtp_coords[1:])
    return baduk.Point(row, col)


def parse_gtp_move(gtp_move):
    if gtp_move.lower() == 'pass':
        return baduk.Move.pass_turn()
    if gtp_move.lower() == 'resign':
        return baduk.Move.resign()
    point = parse_gtp_coords(gtp_move)
    return baduk.Move.play(point)


def parse_gtp_color(color):
    if color.lower().startswith('b'):
        return baduk.Player.black
    if color.lower().startswith('w'):
        return baduk.Player.white
    raise ValueError(color)


def encode_gtp_color(color):
    if color == baduk.Player.black:
        return 'b'
    if color == baduk.Player.white:
        return 'w'
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

    def get_handlers(self):
        known_commands = []
        for attr in dir(self):
            if attr.startswith('handle_'):
                cmd_name = attr[7:]
                known_commands.append(cmd_name)
        known_commands.sort()
        return known_commands

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
        return success('\n'.join(self.get_handlers()))

    def handle_known_command(self, command_name):
        is_known = command_name in self.get_handlers()
        return success('true' if is_known else 'false')

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
        move = parse_gtp_move(coords)
        if player != self.game.next_player:
            # Pretend there was a pass :\
            self.game = self.game.apply_move(baduk.Move.pass_turn())
        self.game = self.game.apply_move(move)
        return success('ok')

    def handle_genmove(self, color):
        player = parse_gtp_color(color)
        if player != self.game.next_player:
            # Pretend there was a pass :\
            self.game = self.game.apply_move(baduk.Move.pass_turn())
        move = self.bot.select_move(self.game)
        self.game = self.game.apply_move(move)
        return success(encode_gtp_move(move))

    def handle_fixed_handicap(self, num_stones):
        num_stones = int(num_stones)
        stones = HANDICAP_STONES[num_stones]
        for gtp_point in stones:
            point = parse_gtp_coords(gtp_point)
            self.board.place_stone(baduk.Player.black, point)
        self.game = baduk.GameState.from_board(
            self.board, baduk.Player.white, self.komi)
        return success('ok')

    def handle_set_free_handicap(self, *stones):
        for gtp_point in stones:
            point = parse_gtp_coords(gtp_point)
            self.board.place_stone(baduk.Player.black, point)
        self.game = baduk.GameState.from_board(
            self.board, baduk.Player.white, self.komi)
        return success('ok')

    def handle_showboard(self):
        return success('')

    def handle_time_settings(self, main, byoyomi, num_stones):
        return success('ok')

    def handle_time_left(self, color, main, num_stones):
        return success('ok')
