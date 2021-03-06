import argparse
import datetime

import baduk
import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game-record-out', '-o')
    parser.add_argument('--black-options', '-b')
    parser.add_argument('--white-options', '-w')
    parser.add_argument('black_bot')
    parser.add_argument('white_bot')
    args = parser.parse_args()

    black_bot = badukai.bots.load_bot(args.black_bot)
    if args.black_options:
        black_options = badukai.options.parse(args.black_options)
        for k, v in black_options.items():
            black_bot.set_option(k, v)
    white_bot = badukai.bots.load_bot(args.white_bot)
    if args.white_options:
        white_options = badukai.options.parse(args.white_options)
        for k, v in white_options.items():
            white_bot.set_option(k, v)
    players = {
        baduk.Player.black: black_bot,
        baduk.Player.white: white_bot,
    }

    assert black_bot.board_size() == white_bot.board_size()
    board_size = black_bot.board_size()

    game = baduk.GameState.new_game(board_size)
    printer = badukai.io.AsciiBoardPrinter()
    while not game.is_over():
        next_bot = players[game.next_player]
        next_move = next_bot.select_move(game)
        game = game.apply_move(next_move)
        printer.print_board(game.board)
        print('')

    game_result = badukai.scoring.compute_game_result(game)
    printer.print_result(game_result)

    if args.game_record_out:
        with open(args.game_record_out, 'w') as game_outf:
            badukai.io.save_game_as_sgf(
                game_outf,
                game, game_result,
                black_name=black_bot.name(),
                white_name=black_bot.name(),
                date=datetime.date.today(),
                komi=7.5)

if __name__ == '__main__':
    main()
