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


def get_module_name(bot):
    path = bot.__module__.split('.')
    assert path[0] == 'badukai'
    assert path[1] == 'bots'
    return path[2]


def save_bot(bot, botfile):
    if isinstance(botfile, six.string_types):
        botfile = h5py.File(botfile, 'w')
    bot_data = botfile.create_group('bot')
    bot_data.attrs['bot_module'] = get_module_name(bot)

    bot_config = bot_data.create_group('config')
    bot.serialize(bot_config)
