import argparse
import os
import uuid
from operator import attrgetter

import requests

import baduk
import badukai


def complete_game(game, black_bot, white_bot, bump_temp_before):
    assert black_bot.board_size() == white_bot.board_size()
    board_size = black_bot.board_size()

    max_game_length = max(
        2 * board_size * board_size,
        65)

    players = {
        baduk.Player.black: black_bot,
        baduk.Player.white: white_bot,
    }

    num_moves = 0
    builder = badukai.bots.zero.GameRecordBuilder()
    printer = badukai.io.AsciiBoardPrinter()
    while not game.is_over():
        next_bot = players[game.next_player]
        if num_moves < bump_temp_before:
            next_bot.set_temperature(1.0)
        else:
            next_bot.set_temperature(0.0)
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
        badukai.kerasutil.set_gpu_memory_target(args.gpu_frac)

    with badukai.io.get_input(args.bot) as bot_file:
        bot = badukai.bots.load_bot(bot_file)
        #white_bot = badukai.bots.load_bot(bot_file)

    if args.options:
        options = badukai.options.parse(args.options)
        for k, v in options.items():
            bot.set_option(k, v)

    game_records = []
    for i in range(args.games):
        print('Game {}/{}...'.format(i + 1, args.games))
        index = badukai.selfplay.load_index(open(args.index))
        worst = index.sample(0.1)
        game = badukai.selfplay.retrieve_game_state(worst)
        print(worst)
        printer = badukai.io.AsciiBoardPrinter()
        printer.print_board(game.board)
        print('Next player: {}'.format(
            printer.format_player(game.next_player)))

        record = complete_game(
            game,
            bot, bot,
            bump_temp_before=2)
        game_records.append(record)

        print('Original position was: {} with {} to play'.format(
            worst,
            printer.format_player(game.next_player)
        ))
    output_file = args.output
    with badukai.io.open_output(output_file) as outf:
        badukai.bots.zero.save_game_records(game_records, outf)


if __name__ == '__main__':
    main()
