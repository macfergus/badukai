import argparse
import datetime
import time

import baduk
import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot', '-b', required=True)
    args = parser.parse_args()

    bot = badukai.bots.load_bot(args.bot)
    num_rollouts = 1200
    bot.set_num_rollouts(num_rollouts)
    board_size = bot.board_size()
    game = baduk.GameState.new_game(board_size)
    bot.set_temperature(1.0)
    start = time.time()
    bot.select_move(game)
    end = time.time()
    elapsed = end - start
    print('%.3fs ==> %.1f per second' % (
        elapsed,
        num_rollouts / elapsed))


if __name__ == '__main__':
    main()
