import argparse
import os
import uuid

import requests

import baduk
import badukai


def record_game(black_bot, white_bot):
    assert black_bot.board_size() == white_bot.board_size()
    board_size = black_bot.board_size()

    max_game_length = 2 * board_size * board_size

    players = {
        baduk.Player.black: black_bot,
        baduk.Player.white: white_bot,
    }

    num_moves = 0
    game = baduk.GameState.new_game(board_size)
    builder = badukai.bots.zero.GameRecordBuilder()
    printer = badukai.io.AsciiBoardPrinter()
    while not game.is_over():
        next_bot = players[game.next_player]
        if num_moves < 20:
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

    game_result = badukai.scoring.compute_game_result(game)
    printer.print_result(game_result)
    builder.record_result(game_result.winner)
    return builder.build()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot', '-b', required=True)
    parser.add_argument('--num-games', '-g', type=int, required=True)
    parser.add_argument('--rollouts-per-move', '-r', type=int, required=True)
    parser.add_argument('--game-record-out', '-o', required=True)
    parser.add_argument('--gpu-frac', type=float)
    args = parser.parse_args()

    if args.gpu_frac is not None:
        badukai.kerasutil.set_gpu_memory_target(args.gpu_frac)

    with badukai.io.get_input(args.bot) as bot_file:
        black_bot = badukai.bots.load_bot(bot_file)
        white_bot = badukai.bots.load_bot(bot_file)

    black_bot.set_num_rollouts(args.rollouts_per_move)
    white_bot.set_num_rollouts(args.rollouts_per_move)

    board_size = black_bot.board_size()

    game_records = []
    for i in range(args.num_games):
        print('Game %d/%d...' % (i + 1, args.num_games))
        game_records.append(record_game(black_bot, white_bot))

    output_file = args.game_record_out
    if os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX'):
        output_file += '-' + str(uuid.uuid4())
    with badukai.io.open_output(output_file) as outf:
        badukai.bots.zero.save_game_records(game_records, outf)


if __name__ == '__main__':
    main()
