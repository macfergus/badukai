import importlib

__all__ = [
    'get_network_by_name',
]


def get_network_by_name(name, input_shape):
    module = importlib.import_module('badukai.networks.' + name)
    constructor = getattr(module, 'layers')
    return constructor(input_shape)
