import argparse
import boto3
import io
import os
import tempfile

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('inputs', nargs='+')
    args = parser.parse_args()

    records = []
    for input_file in args.inputs:
        print('Reading {}...'.format(input_file))
        records += badukai.bots.zero.load_game_records(open(input_file))

    with badukai.io.open_output(args.output) as outf:
        badukai.bots.zero.save_game_records(records, outf)


if __name__ == '__main__':
    main()
