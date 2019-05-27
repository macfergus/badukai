import argparse
import datetime
import enum
import os
import random
import time
from collections import namedtuple
from operator import attrgetter

import requests

import baduk
import badukai


def complete_game(bot, pachi_client, bot_player):
    bump_temp_before = 10

    game = baduk.GameState.new_game(19)
    printer = badukai.io.AsciiBoardPrinter()
    num_moves = 0
    while not game.is_over():
        if num_moves < bump_temp_before:
            bot.set_temperature(1.0)
        else:
            bot.set_temperature(0.0)
        if game.next_player == bot_player:
            next_move = bot.select_move(game)
            pachi_client.send_move(game.next_player, next_move)
        else:
            next_move = pachi_client.select_move(game)
        game = game.apply_move(next_move)
        num_moves += 1
        printer.print_board(game.board)
        print('')

    game_result = badukai.scoring.remove_dead_stones_and_score(game)
    print('GAME OVER')
    printer.print_board(game_result.final_board)
    printer.print_result(game_result)
    if game_result.winner == bot_player:
        print('Bot wins!')
    else:
        print('Pachi wins')
    return game, game_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot', '-b', required=True)
    parser.add_argument('--options')
    parser.add_argument('--pachi', '-p', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--wins', '-w')
    parser.add_argument('--log')
    args = parser.parse_args()

    with badukai.io.get_input(args.bot) as bot_file:
        bot = badukai.bots.load_bot(bot_file)

    if args.options:
        options = badukai.options.parse(args.options)
        for k, v in options.items():
            bot.set_option(k, v)

    if args.log:
        logf = open(args.log, 'a')
    else:
        logf = None

    if logf is not None:
        logf.write('***\n')
        logf.write('Using bot: {}\n'.format(args.bot))
        logf.write('Bot settings: {}\n'.format(args.options))
        logf.write('Pachi settings: {}\n'.format(args.pachi))
        logf.write('***\n')

    num_games = 0
    losses = 0
    start = time.time()
    while True:
        pachi = badukai.gtp.GTPClient(args.pachi)
        bot_player = random.choice([baduk.Player.black, baduk.Player.white])
        try:
            game, game_result = complete_game(bot, pachi, bot_player)
            num_games += 1
            fname = 'pachi_{}.sgf'.format(int(time.time()))
            if game_result.winner != bot_player:
                losses += 1
                path = os.path.join(args.output, fname)
                with open(path, 'w') as outf:
                    badukai.io.save_game_as_sgf(
                        outf,
                        game,
                        str(game_result),
                        black_name='Bot' if bot_player == baduk.Player.black else 'Pachi',
                        white_name='Bot' if bot_player == baduk.Player.white else 'Pachi',
                        date=datetime.datetime.now(),
                        komi=7.5)
            elif args.wins:
                path = os.path.join(args.wins, fname)
                with open(path, 'w') as outf:
                    badukai.io.save_game_as_sgf(
                        outf,
                        game,
                        str(game_result),
                        black_name='Bot' if bot_player == baduk.Player.black else 'Pachi',
                        white_name='Bot' if bot_player == baduk.Player.white else 'Pachi',
                        date=datetime.datetime.now(),
                        komi=7.5)

        finally:
            pachi.shutdown()
        wins = num_games - losses
        print('***')
        print('***')
        print('*** completed {} games'.format(num_games))
        print('*** record: {} / {}'.format(wins, num_games))
        print('***')
        print('***')
        if logf is not None:
            elapsed = time.time() - start
            rate = num_games / (elapsed / 3600.0)
            logf.write('Completed {} games in {} seconds ({:.1f} per hour)\n'.format(
                num_games,
                int(elapsed),
                rate))
            logf.write('Bot record: {} / {} ({:.3f})\n'.format(
                wins, num_games,
                float(wins) / float(num_games)))
            logf.flush()


if __name__ == '__main__':
    main()
