__all__ = [
    'parse',
]


def parse(options_str):
    pairs = options_str.split(',')
    options = {}
    for pair in pairs:
        k, v = pair.split('=')
        options[k] = v
    return options
