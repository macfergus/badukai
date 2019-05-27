import argparse
import random
import time
from collections import namedtuple

import h5py
import numpy as np


Datasource = namedtuple('Datasource', 'X action value n')

def open_dataset(fname):
    h5file = h5py.File(fname, 'r')
    n = h5file['X'].len()
    return Datasource(
        X=h5file['X'],
        action=h5file['y_action'],
        value=h5file['y_value'],
        n=n)


def h5_append(dataset, array):
    assert dataset.shape[1:] == array.shape[1:]
    n = dataset.len()
    extra = array.shape[0]
    dataset.resize((n + extra,) + dataset.shape[1:])
    dataset[n:] = array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', '-p', required=True)
    parser.add_argument('--input', '-i', action='append')
    parser.add_argument('--ratio', '-r', type=float, action='append')
    parser.add_argument('--output', '-o', required=True)
    args = parser.parse_args()

    assert len(args.input) == len(args.ratio)
    print(args.input)
    print(args.ratio)

    datasets = {}
    primary = open_dataset(args.primary)
    datasets['primary'] = primary
    indices = [('primary', i) for i in range(primary.n)]

    for fname, ratio in zip(args.input, args.ratio):
        print('open {}'.format(fname))
        datasets[fname] = open_dataset(fname)
        target_size = int(primary.n / ratio)
        print('Take {} records from {}'.format(target_size, fname))
        idx = np.random.choice(datasets[fname].n, size=target_size, replace=False)
        indices += [(fname, i) for i in idx]
    random.shuffle(indices)

    output_shape = (len(indices),) + primary.X.shape[1:]
    action_shape = (len(indices),) + primary.action.shape[1:]
    value_shape = (len(indices),) + primary.value.shape[1:]

    print(output_shape, action_shape, value_shape)

    with h5py.File(args.output, 'w') as h5out:
        X_out = h5out.create_dataset('X', shape=output_shape)
        action_out = h5out.create_dataset('y_action', shape=action_shape)
        value_out = h5out.create_dataset('y_value', shape=value_shape)

        start = time.time()
        n = len(indices)
        for j, pair in enumerate(indices):
            key, i = pair
            src = datasets[key]
            X_out[j] = src.X[i]
            action_out[j] = src.action[i]
            value_out[j] = src.value[i]
            if j > 1 and j % 1000 == 0:
                elapsed = time.time() - start
                remaining = n - j
                rate = float(j) / elapsed
                eta = remaining / rate
                print('{} / {} (eta {:.1f})'.format(j, n, eta))


if __name__ == '__main__':
    main()
