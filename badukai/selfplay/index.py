import json

from ..io import read_game_from_sgf

__all__ = [
    'build_index',
]


class Position:
    def __init__(self, source_file, move_num, state, error):
        self.source_file = source_file
        self.move_num = move_num
        self.state = state
        self.error = error


def extract_positions(bot, sgf_file):
    game_record = read_game_from_sgf(open(sgf_file))
    state = game_record.initial_state
    winner = game_record.winner
    positions = []
    for i, move in enumerate(game_record.moves):
        state = state.apply_move(move)
        if state.next_player != winner:
            value = bot.evaluate(state)
            error = value - (-1)
            positions.append(Position(
                sgf_file, i, state, error))
    return positions


class LossIndex:
    def __init__(self, positions):
        self.positions = positions

    def serialize(self, outf):
        positions_json = [{
            'source_file': pos.source_file,
            'move_num': pos.move_num,
            'error': pos.error,
        } for pos in self.positions]
        json.dump(positions_json, outf)


def build_index(bot, sgf_files):
    positions = []
    for i, sgf_filename in enumerate(sgf_files):
        print('Processing {} (game {})...'.format(sgf_filename, i + 1))
        positions += extract_positions(bot, sgf_filename)
    return LossIndex(positions)
