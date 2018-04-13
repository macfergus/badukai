import json
from collections import namedtuple

from baduk import Player

__all__ = [
    'GameRecordBuilder',
    'GameRecord',
    'save_game_records',
]


class MoveRecord(namedtuple('MoveRecord', 'player move visit_counts')):
    pass


class GameRecord(object):
    def __init__(self, move_records, winner):
        self.move_records = list(move_records)
        self.winner = winner


class GameRecordBuilder(object):
    def __init__(self):
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
        return GameRecord(self._move_records, self._winner)


class GameRecordSerializer(object):
    def __init__(self, outf):
        self._out = outf

    def write_game_records(self, game_records):
        self.write_int(len(game_records))
        for record in game_records:
            self.write_game_record(record)
        self.write_char('\n')

    def write_game_record(self, game_record):
        self.write_int(len(game_record.move_records))
        for move_record in game_record.move_records:
            self.write_move_record(move_record)
        self.write_player(game_record.winner)

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
        for digit in str(i):
            self.write_char(digit)
        self.write_char('.')

    def write_bool(self, b):
        self.write_char('t' if b else 'f')

    def write_char(self, c):
        self._out.write(c)


def save_game_records(records, outf):
    serializer = GameRecordSerializer(outf)
    serializer.write_game_records(records)
