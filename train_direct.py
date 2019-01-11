import argparse

import h5py
import numpy as np

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--chunk-size', '-c', type=int)
    parser.add_argument('--bot', required=True)
    parser.add_argument('--bot-out', required=True)
    args = parser.parse_args()

    with badukai.io.get_input(args.bot) as bot_file:
        bot = badukai.bots.load_bot(bot_file)

    with h5py.File(args.input, 'r') as h5in:
        X = h5in['X']
        y_action = h5in['y_action']
        y_value = h5in['y_value']
        n = X.len()
        print('{} total observations'.format(n))
        assert y_action.len() == n
        assert y_value.len() == n
        n_chunk = n // args.chunk_size
        if n_chunk * args.chunk_size < n:
            n_chunk += 1
        for i in range(n_chunk):
            print('{} / {}...'.format(i + 1, n_chunk))
            chunk_start = i * args.chunk_size
            chunk_end = chunk_start + args.chunk_size
            batch_X = np.array(X[chunk_start:chunk_end])
            batch_action = np.array(y_action[chunk_start:chunk_end])
            batch_value = np.array(y_value[chunk_start:chunk_end])
            bot.train_direct(batch_X, batch_action, batch_value)

    with badukai.io.open_output_filename(args.bot_out) as output_filename:
        badukai.bots.save_bot(bot, output_filename)


if __name__ == '__main__':
    main()
