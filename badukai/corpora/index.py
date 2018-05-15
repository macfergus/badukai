import json
from io import StringIO

from .archive import SGFLocator, find_sgfs, tarball_iterator
from ..io import read_game_from_sgf

__all__ = [
    'build_index',
    'load_checkpoint',
    'open_index',
]


class Pointer:
    def __init__(self, epoch, index):
        self.epoch = epoch
        self.idx = index

    def __str__(self):
        return '{}.{}'.format(self.epoch, self.idx)

    def save(self, outf):
        json.dump({
            'epoch': self.epoch,
            'index': self.idx,
        }, outf)


def load_checkpoint(inf):
    data = json.load(inf)
    return Pointer(data['epoch'], data['index'])


class Corpus:
    def __init__(self, physical_files, boundaries):
        self.physical_files = physical_files
        self.boundaries = boundaries

    def start(self):
        return Pointer(0, 0)

    def next(self, pointer):
        epoch = pointer.epoch
        next_idx = pointer.idx + 1
        if next_idx >= len(self.boundaries):
            next_idx = 0
            epoch += 1
        return Pointer(epoch, next_idx)

    def get_game_records(self, pointer):
        start = self.boundaries[pointer.idx]
        end = None
        if pointer.idx < len(self.boundaries) - 1:
            end = self.boundaries[pointer.idx + 1]
        idx = self.physical_files.index(start.physical_file)
        done = False
        game_records = []
        while idx < len(self.physical_files):
            physical_file = self.physical_files[idx]
            with tarball_iterator(physical_file) as tarball:
                for sgf in tarball:
                    if sgf.locator < start:
                        continue
                    if end is not None and not (sgf.locator < end):
                        done = True
                        break
                    try:
                        f = StringIO(sgf.contents)
                        game = read_game_from_sgf(f)
                        if abs(game.initial_state.komi()) > 9:
                            # print('Reject {}: komi is {}'.format(
                            #    sgf, game.initial_state.komi()))
                            pass
                        else:
                            game_records.append(game)
                    except KeyError as e:
                        # print('Reject {}: missing {}'.format(sgf, e))
                        pass
                    except ValueError as e:
                        print('Error on {}: {}'.format(sgf, e))
                        pass
            idx += 1
        return game_records

    def serialize(self, outf):
        boundaries = [boundary.to_json() for boundary in self.boundaries]
        json.dump({
            'physical_files': self.physical_files,
            'boundaries': boundaries,
        }, outf)


def build_index(path, chunk_size):
    physical_files = set()
    boundaries = []
    first = True
    games = 0
    for sgf in find_sgfs(path):
        physical_files.add(sgf.locator.physical_file)
        games += 1
        if first or games == chunk_size:
            boundaries.append(sgf.locator)
            games = 0
            first = False
    return Corpus(list(sorted(physical_files)), boundaries)


def open_index(inf):
    data = json.load(inf)
    boundaries = [
        SGFLocator.from_json(boundary)
        for boundary in data['boundaries']]
    return Corpus(data['physical_files'], boundaries)
