import argparse
import random
import time

import h5py
import numpy as np


def h5_append(dataset, array):
    assert dataset.shape[1:] == array.shape[1:]
    n = dataset.len()
    extra = array.shape[0]
    dataset.resize((n + extra,) + dataset.shape[1:])
    dataset[n:] = array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--ratio', '-r', type=float)
    parser.add_argument('self_play_file')
    parser.add_argument('human_file')
    args = parser.parse_args()

    self_play_in = h5py.File(args.self_play_file, 'r')
    human_in = h5py.File(args.human_file, 'r')
    n_self_play = self_play_in['X'].len()
    n_human = human_in['X'].len()
    target_n_human = int(n_self_play / args.ratio)

    self_idx = np.arange(n_self_play)
    human_idx = np.random.choice(n_human, size=target_n_human, replace=False)
    indices = [('self', i) for i in self_idx]
    indices += [('human', i) for i in human_idx]
    random.shuffle(indices)

    self_X = self_play_in['X']
    self_action = self_play_in['y_action']
    self_value = self_play_in['y_value']
    human_X = human_in['X']
    human_action = human_in['y_action']
    human_value = human_in['y_value']

    output_shape = (len(indices),) + self_X.shape[1:]
    action_shape = (len(indices),) + self_action.shape[1:]
    value_shape = (len(indices),) + self_value.shape[1:]

    print(output_shape, action_shape, value_shape)

    with h5py.File(args.output, 'w') as h5out:
        X_out = h5out.create_dataset('X', shape=output_shape)
        action_out = h5out.create_dataset('y_action', shape=action_shape)
        value_out = h5out.create_dataset('y_value', shape=value_shape)

        start = time.time()
        n = len(indices)
        for j, pair in enumerate(indices):
            src, i = pair
            if src == 'self':
                X_out[j] = self_X[i]
                action_out[j] = self_action[i]
                value_out[j] = self_value[i]
            else:
                X_out[j] = human_X[i]
                action_out[j] = human_action[i]
                value_out[j] = human_value[i]
            if j > 1 and j % 1000 == 0:
                elapsed = time.time() - start
                remaining = n - j
                rate = float(j) / elapsed
                eta = remaining / rate
                print('{} / {} (eta {:.1f})'.format(j, n, eta))


if __name__ == '__main__':
    main()
