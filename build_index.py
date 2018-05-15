import argparse

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-size', type=int, default=5000)
    parser.add_argument('path')
    parser.add_argument('output_file')
    args = parser.parse_args()

    corpus = badukai.corpora.build_index(args.path, args.chunk_size)
    with open(args.output_file, 'w') as outf:
        corpus.serialize(outf)


if __name__ == '__main__':
    main()
