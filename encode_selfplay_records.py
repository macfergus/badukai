import argparse

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
    parser.add_argument('game_records', nargs='+')
    args = parser.parse_args()

    with badukai.io.get_input(args.bot) as bot_file:
        bot = badukai.bots.load_bot(bot_file)

    num_files = len(args.game_records)
    total_records = 0
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

        for i, fname in enumerate(args.game_records):
            print('Processing {} ({}/{})...'.format(
                fname,
                i + 1, num_files))
            with badukai.io.get_input(fname) as games_file:
                game_records = list(badukai.bots.zero.generate_game_records(
                    open(games_file)))
                for game in game_records:
                    next_X, next_action, next_value = bot.encode_game(game)
                    if next_X.size == 0:
                        continue
                    h5_append(X, next_X)
                    h5_append(y_action, next_action)
                    h5_append(y_value, next_value)
                    total_records += 1
                print('{} records'.format(total_records))


if __name__ == '__main__':
    main()
