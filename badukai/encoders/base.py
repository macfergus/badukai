import importlib

__all__ = [
    'Encoder',
    'get_encoder_by_name',
    'load_encoder',
    'save_encoder',
]


class Encoder(object):
    def board_size(self):
        raise NotImplementedError()

    def encode(self, game_state):
        raise NotImplementedError()

    def encode_move(self, move):
        raise NotImplementedError()

    def decode_index(self, index):
        raise NotImplementedError()

    def num_moves(self):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()


def get_encoder_by_name(name, board_size):
    module = importlib.import_module('badukai.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size)


def save_encoder(encoder, h5group):
    encoder_module_name = encoder.__module__.split('.')[-1]
    h5group.attrs['name'] = encoder_module_name
    h5group.attrs['board_size'] = encoder.board_size()


def load_encoder(h5group):
    return get_encoder_by_name(
        h5group.attrs['name'],
        h5group.attrs['board_size'])
