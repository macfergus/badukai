import argparse
import time

import baduk
import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', '-o')
    parser.add_argument('bot')
    args = parser.parse_args()

    bot = badukai.bots.load_bot(args.bot)
    if args.options:
        options = badukai.options.parse(args.options)
        for k, v in options.items():
            bot.set_option(k, v)
    num_rollouts = 256
    bot.set_option('num_rollouts', num_rollouts)

    board_size = bot.board_size()
    game = baduk.GameState.new_game(board_size)
    start = time.time()
    next_move = bot.select_move(game)
    end = time.time()
    elapsed = end - start
    per_sec = float(num_rollouts) / elapsed
    print('{} rollouts per second'.format(per_sec))

if __name__ == '__main__':
    main()
