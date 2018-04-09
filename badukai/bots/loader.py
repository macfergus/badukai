import importlib

import h5py
import six

__all__ = [
    'load_bot',
    'save_bot',
]


def get_ctor_by_name(module_name):
    full_module_path = 'badukai.bots.' + module_name
    mod = importlib.import_module(full_module_path)
    return mod.load_from_hdf5


def load_bot(botfile):
    if isinstance(botfile, six.string_types):
        botfile = h5py.File(botfile, 'r')
    bot_data = botfile['bot']
    bot_module_name = bot_data.attrs['bot_module']

    ctor = get_ctor_by_name(bot_module_name)
    return ctor(bot_data['config'])


def save_bot(bot, botfile):
    if isinstance(botfile, six.string_types):
        botfile = h5py.File(botfile, 'w')
    bot_module_name = bot.__module__.split('.')[-1]
    bot_data = botfile.create_group('bot')
    bot_data.attrs['bot_module'] = bot_module_name

    bot_config = bot_data.create_group('config')
    bot.serialize(bot_config)
