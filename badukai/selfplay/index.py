import json
import numpy as np
import os
import time

from ..io import read_game_from_sgf

__all__ = [
    'build_index',
    'load_index',
    'retrieve_game_state',
]


def get_create_ts(filename):
    return os.stat(filename).st_ctime


class Position:
    def __init__(self, source_file, move_num, error, create_ts):
        self.source_file = source_file
        self.move_num = move_num
        self.error = error
        self.create_ts = create_ts

    def __str__(self):
        return '{}:{} ({})'.format(self.source_file, self.move_num, self.error)


def extract_positions(bot, sgf_file):
    create_ts = get_create_ts(sgf_file)
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
                sgf_file, i, error, create_ts))
    return positions


class LossIndex:
    def __init__(self, positions):
        self.positions = positions
        self._unique_files = {pos.source_file for pos in self.positions}

    def serialize(self, outf):
        positions_json = [{
            'source_file': pos.source_file,
            'move_num': pos.move_num,
            'error': pos.error,
            'create_ts': pos.create_ts,
        } for pos in self.positions]
        json.dump(positions_json, outf)

    def sample(self, temperature=1.0, decay_per_day=0.0):
        now = time.time()
        creates = np.array([pos.create_ts for pos in self.positions])
        ages_s = now - creates
        ages_days = ages_s / (24 * 3600.0)
        age_decays = np.power((1 - decay_per_day), ages_days)
        p = np.array([pos.error for pos in self.positions]) * age_decays + 1e-4
        p /= np.sum(p)
        p = np.power(p, 1.0 / temperature)
        p /= np.sum(p)
        return np.random.choice(self.positions, p=p)

    def update(self, bot, sgf_file):
        try:
            new_positions = extract_positions(bot, sgf_file)
        except KeyError as e:
            print('Could not process {}: {}'.format(sgf_file, e))
            return
        for new_pos in new_positions:
            self.positions.append(new_pos)
            self._unique_files.add(new_pos.source_file)


    def __contains__(self, fname):
        return fname in self._unique_files


def build_index(bot, sgf_files):
    positions = []
    for i, sgf_filename in enumerate(sgf_files):
        print('Processing {} (game {})...'.format(sgf_filename, i + 1))
        try:
            positions += extract_positions(bot, sgf_filename)
        except KeyError as e:
            print('Could not process {}: {}'.format(sgf_filename, e))
    return LossIndex(positions)


def load_index(indexfile):
    positions_json = json.load(indexfile)
    positions = [Position(
        pos['source_file'],
        pos['move_num'],
        pos['error'],
        pos['create_ts'],
    ) for pos in positions_json]
    return LossIndex(positions)


def retrieve_game_state(position):
    record = read_game_from_sgf(open(position.source_file))
    game = record.initial_state
    for i in range(position.move_num + 1):
        game = game.apply_move(record.moves[i])
    return game
