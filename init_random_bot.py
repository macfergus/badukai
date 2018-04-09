import argparse

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=19)
    parser.add_argument('botfile')
    args = parser.parse_args()

    random_bot = badukai.bots.RandomBot(args.board_size)
    badukai.bots.save_bot(random_bot, args.botfile)


if __name__ == '__main__':
    main()
