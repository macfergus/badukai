import argparse
import os

import numpy as np

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bot')
    parser.add_argument('game_records')
    parser.add_argument('index_file')
    args = parser.parse_args()

    bot = badukai.bots.load_bot(args.bot)

    sgf_files = []
    for dirpath, dirnames, names in os.walk(args.game_records):
        for fname in names:
            full_path = os.path.join(dirpath, fname)
            if full_path.endswith('.sgf'):
                sgf_files.append(full_path)

    index = badukai.selfplay.build_index(bot, sgf_files)
    with open(args.index_file, 'w') as outf:
        index.serialize(outf)


if __name__ == '__main__':
    main()
