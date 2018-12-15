import argparse
import os

import h5py

import badukai


def h5_append(dataset, array):
    assert dataset.shape[1:] == array.shape[1:]
    n = dataset.len()
    extra = array.shape[0]
    dataset.resize((n + extra,) + dataset.shape[1:])
    dataset[n:] = array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('index_file')
    args = parser.parse_args()

    with badukai.io.get_input(args.bot) as bot_file:
        bot = badukai.bots.load_bot(bot_file)

    corpus = badukai.corpora.open_index(open(args.index_file))
    chunk = corpus.start()

    with h5py.File(args.output, 'w') as h5out:
        X = h5out.create_dataset(
            'X',
            shape=(0,) + bot._encoder.shape(),
            maxshape=(None,) + bot._encoder.shape())
        y_action = h5out.create_dataset(
            'y_action',
            shape=(0, bot._encoder.num_moves()),
            maxshape=(None, bot._encoder.num_moves()))
        y_value = h5out.create_dataset(
            'y_value',
            shape=(0,),
            maxshape=(None,))

        while chunk is not None:
            print('Loading chunk {}...'.format(chunk))
            game_records = corpus.get_game_records(chunk)
            print('Chunk {} has {} usable game records'.format(
                chunk, len(game_records)))
            next_X, next_action, next_value = bot.encode_human(game_records)
            h5_append(X, next_X)
            h5_append(y_action, next_action)
            h5_append(y_value, next_value)

            chunk = corpus.next(chunk, wrap=False)


if __name__ == '__main__':
    main()
