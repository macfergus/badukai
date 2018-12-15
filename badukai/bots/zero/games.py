import json
from collections import namedtuple

from baduk import Board, GameState, Move, Player, Point

__all__ = [
    'GameRecordBuilder',
    'GameRecord',
    'load_game_records',
    'generate_game_records',
    'save_game_records',
]


class MoveRecord(namedtuple('MoveRecord', 'player move visit_counts')):
    pass


class GameRecord(object):
    def __init__(self, initial_state, move_records, winner):
        self.initial_state = initial_state
        self.move_records = list(move_records)
        self.winner = winner


class GameRecordBuilder(object):
    def __init__(self, initial_state):
        self._initial_state = initial_state
        self._move_records = []
        self._winner = None

    def record_move(self, player, move, visit_counts):
        self._move_records.append(
            MoveRecord(
                player=player,
                move=move,
                visit_counts=visit_counts))

    def record_result(self, winner):
        self._winner = winner

    def build(self):
        assert self._winner is not None
        return GameRecord(
            self._initial_state,
            self._move_records,
            self._winner)


class GameRecordSerializer(object):
    def __init__(self, outf):
        self._out = outf

    def write_game_state(self, game_state):
        states = []
        while game_state is not None:
            states.append(game_state)
            game_state = game_state.previous_state
        states.reverse()

        empty_state = states[0]
        init_board = empty_state.board
        self.write_board(init_board)
        self.write_player(empty_state.next_player)
        self.write_float(states[0].komi())

        other_states = states[1:]
        for state in other_states:
            self.write_optional(state.last_move, self.write_move)
        self.write_optional(None, self.write_move)

    def write_board(self, board):
        assert board.num_cols == board.num_rows
        self.write_int(board.num_rows)
        for r in range(1, board.num_rows + 1):
            for c in range(1, board.num_cols + 1):
                p = Point(r, c)
                self.write_optional(board.get(p), self.write_player)

    def write_optional(self, value, write_fun):
        self.write_bool(value is None)
        if value is not None:
            write_fun(value)

    def write_game_records(self, game_records):
        self.write_int(len(game_records))
        for record in game_records:
            self.write_game_record(record)

    def write_game_record(self, game_record):
        self.write_game_state(game_record.initial_state)
        self.write_int(len(game_record.move_records))
        moves = []
        for move_record in game_record.move_records:
            self.write_move_record(move_record)
        self.write_player(game_record.winner)
        self.write_char('\n')

    def write_player(self, player):
        assert player in (Player.black, Player.white)
        self.write_char('B' if player == Player.black else 'W')

    def write_move_record(self, move_record):
        self.write_player(move_record.player)
        self.write_move(move_record.move)
        self.write_visit_counts(move_record.visit_counts)

    def write_move(self, move):
        self.write_bool(move.is_play)
        self.write_bool(move.is_pass)
        self.write_bool(move.is_resign)
        if move.is_play:
            self.write_int(move.point.row)
            self.write_int(move.point.col)

    def write_visit_counts(self, visit_counts):
        self.write_int(len(visit_counts))
        for c in visit_counts:
            self.write_int(c)

    def write_int(self, i):
        for digit in str(int(i)):
            self.write_char(digit)
        self.write_char('.')

    def write_float(self, f):
        for digit in str(f):
            self.write_char(digit)
        self.write_char(';')

    def write_bool(self, b):
        self.write_char('t' if b else 'f')

    def write_char(self, c):
        self._out.write(c)


class GameRecordDeserializer(object):
    def __init__(self, inf):
        self._in = inf

    def read_game_records(self):
        num_game_records = self.read_int()
        records = []
        for _ in range(num_game_records):
            records.append(self.read_game_record())
        return records

    def generate_game_records(self):
        num_game_records = self.read_int()
        for _ in range(num_game_records):
            yield self.read_game_record()

    def read_game_record(self):
        initial_state = self.read_game_state()
        num_moves = self.read_int()
        move_records = []
        for _ in range(num_moves):
            move_records.append(self.read_move_record())
        winner = self.read_player()
        assert self.read_char() == '\n'
        return GameRecord(
            initial_state=initial_state,
            move_records=move_records,
            winner=winner)

    def read_game_state(self):
        board = self.read_board()
        next_player = self.read_player()
        komi = self.read_float()
        game = GameState.from_board(board, next_player, komi)
        move = self.read_optional(self.read_move)
        while move is not None:
            game = game.apply_move(move)
            move = self.read_optional(self.read_move)
        return game

    def read_board(self):
        board_size = self.read_int()
        board = Board(board_size, board_size)
        for r in range(1, board.num_rows + 1):
            for c in range(1, board.num_cols + 1):
                p = Point(r, c)
                stone = self.read_optional(self.read_player)
                if stone is not None:
                    board.place_stone(stone, p)
        return board

    def read_optional(self, read_fun):
        is_none = self.read_bool()
        if is_none:
            return None
        return read_fun()

    def read_player(self):
        ch = self.read_char()
        assert ch in ('B', 'W')
        return Player.black if ch == 'B' else Player.white

    def read_move_record(self):
        player = self.read_player()
        move = self.read_move()
        visit_counts = self.read_visit_counts()
        return MoveRecord(
            player=player,
            move=move,
            visit_counts=visit_counts)

    def read_move(self):
        is_play = self.read_bool()
        is_pass = self.read_bool()
        is_resign = self.read_bool()
        if is_play:
            row = self.read_int()
            col = self.read_int()
            return Move.play(Point(row=row, col=col))
        if is_pass:
            return Move.pass_turn()
        assert is_resign
        return Move.resign()

    def read_visit_counts(self):
        num_visit_counts = self.read_int()
        visit_counts = []
        for _ in range(num_visit_counts):
            visit_counts.append(self.read_int())
        return visit_counts

    def read_int(self):
        number_str = ''
        while True:
            c = self.read_char()
            if c == '.':
                break
            assert c.isdigit()
            number_str += c
        return int(number_str)

    def read_float(self):
        number_str = ''
        while True:
            c = self.read_char()
            if c == ';':
                break
            assert c.isdigit() or c == '.'
            number_str += c
        return float(number_str)

    def read_bool(self):
        c = self.read_char()
        assert c in ('t', 'f')
        return c == 't'

    def read_char(self):
        return self._in.read(1)


def save_game_records(records, outf):
    serializer = GameRecordSerializer(outf)
    serializer.write_game_records(records)


def load_game_records(inf):
    deserializer = GameRecordDeserializer(inf)
    return deserializer.read_game_records()

def generate_game_records(inf):
    deserializer = GameRecordDeserializer(inf)
    return deserializer.generate_game_records()
