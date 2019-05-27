import argparse
import datetime
import enum
import os
import random
import uuid
from collections import namedtuple
from operator import attrgetter

import requests

import baduk
import badukai


class ResignLevel(enum.Enum):
    paranoid = 0
    conservative = 1
    aggressive = 2


def choose_resign_level():
    x = random.random()
    if x < 0.15:
        return ResignLevel.paranoid
    if x < 0.25:
        return ResignLevel.aggressive
    return ResignLevel.conservative


class GameInfo(namedtuple('GameInfo', 'orig_pos orig_player resign_level eta_msg')):
    pass


def print_game_info(game_info):
    print('Original game: {}, {} to play'.format(game_info.orig_pos, game_info.orig_player))
    print('Resign level: {}'.format(game_info.resign_level))
    print(game_info.eta_msg)


def get_resign_thresh(resign_level):
    if resign_level == ResignLevel.paranoid:
        # Don't resign, always complete the game
        return -2
    if resign_level == ResignLevel.conservative:
        return -0.97
    if resign_level == ResignLevel.aggressive:
        return -0.8
    raise ValueError(resign_level)


def complete_game(game, black_bot, white_bot,
                  bump_temp_before=0,
                  min_moves_before_resign=9999,
                  resign_below=-2.0,
                  game_info=None):
    assert black_bot.board_size() == white_bot.board_size()
    board_size = black_bot.board_size()

    builder = badukai.bots.zero.GameRecordBuilder(game)

    max_game_length = max(
        2 * board_size * board_size,
        65)

    players = {
        baduk.Player.black: black_bot,
        baduk.Player.white: white_bot,
    }

    num_moves = 0
    printer = badukai.io.AsciiBoardPrinter()
    while not game.is_over():
        next_bot = players[game.next_player]
        if num_moves < bump_temp_before:
            next_bot.set_temperature(1.0)
        else:
            next_bot.set_temperature(0.0)
        if num_moves < min_moves_before_resign:
            next_bot.set_option('resign_below', -2)
        else:
            next_bot.set_option('resign_below', resign_below)
        if num_moves >= max_game_length:
            next_move = baduk.Move.pass_turn()
        else:
            next_move = next_bot.select_move(game)
            builder.record_move(
                game.next_player,
                next_move,
                next_bot.search_counts())
        num_moves += 1
        game = game.apply_move(next_move)
        printer.print_board(game.board)
        if game_info:
            print_game_info(game_info)
        print('')

    game_result = badukai.scoring.remove_dead_stones_and_score(game)
    print('GAME OVER')
    printer.print_board(game_result.final_board)
    printer.print_result(game_result)
    builder.record_result(game_result.winner)
    return builder.build()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot', '-b', required=True)
    parser.add_argument('--index', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--gpu-frac', type=float)
    parser.add_argument('--games', '-g', type=int, default=1)
    parser.add_argument('--options')
    args = parser.parse_args()

    if args.gpu_frac is not None:
        badukai.kerasutil.set_tf_options(gpu_frac=args.gpu_frac)

    with badukai.io.get_input(args.bot) as bot_file:
        bot = badukai.bots.load_bot(bot_file)
        #white_bot = badukai.bots.load_bot(bot_file)

    if args.options:
        options = badukai.options.parse(args.options)
        for k, v in options.items():
            bot.set_option(k, v)

    num_chunks = 0
    game_records = []
    eta_msg = 'No ETA yet'
    start = datetime.datetime.now()
    for i in range(args.games):
        print('Game {}/{}...'.format(i + 1, args.games))
        index = badukai.selfplay.load_index(open(args.index))
        worst = index.sample(1.0 / 10.0, decay_per_day=0.02)
        game = badukai.selfplay.retrieve_game_state(worst)
        game = badukai.symmetry.rotate_game_record(game, random.randint(0, 7))
        print(worst)
        printer = badukai.io.AsciiBoardPrinter()
        printer.print_board(game.board)
        print('Next player: {}'.format(
            printer.format_player(game.next_player)))

        # Vary resignation settings.
        resign_level = choose_resign_level()
        resign_thresh = get_resign_thresh(resign_level)
        print('Resign settings: {} ({:.3f})'.format(resign_level, resign_thresh))
        game_info = GameInfo(
            orig_pos=worst,
            orig_player=game.next_player,
            resign_level=resign_level,
            eta_msg=eta_msg)
        record = complete_game(
            game,
            bot, bot,
            bump_temp_before=4,
            min_moves_before_resign=30,
            resign_below=resign_thresh,
            game_info=game_info)
        game_records.append(record)

        print('Original position was: {} with {} to play'.format(
            worst,
            printer.format_player(game.next_player)
        ))

        if len(game_records) >= 25:
            output_file = '{}_{:03d}'.format(args.output, num_chunks)
            with badukai.io.open_output(output_file) as outf:
                badukai.bots.zero.save_game_records(game_records, outf)
            game_records = []
            num_chunks += 1
            # Refresh the index every time we flush.
            index = badukai.selfplay.load_index(open(args.index))

        end = datetime.datetime.now()
        elapsed_s = (end - start).total_seconds()
        games_per_s = float(i + 1) / elapsed_s
        games_per_hour = 3600.0 * games_per_s
        games_remaining = args.games - i - 1
        if games_remaining > 0:
            eta_s = games_remaining / games_per_s
            eta_dt = end + datetime.timedelta(seconds=eta_s)
            eta_msg = 'Completing {:.1f} games per hour; ETA {}'.format(
                games_per_hour,
                eta_dt.strftime('%Y-%m-%d %H:%M'))

    if game_records:
        output_file = '{}_{:03d}'.format(args.output, num_chunks)
        with badukai.io.open_output(output_file) as outf:
            badukai.bots.zero.save_game_records(game_records, outf)


if __name__ == '__main__':
    main()
