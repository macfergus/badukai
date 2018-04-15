import argparse
import boto3
import datetime
import os
import tempfile
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


def get_bot_from_url(url):
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-httpbot')
    resp = requests.get(url)
    os.write(tempfd, resp.content)
    return tempfname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-input')
    parser.add_argument('--s3-output')
    parser.add_argument('--output', '-o')
    args = parser.parse_args()

    clean_up_local_output = False
    local_output = args.output
    if local_output is None:
        tempfd, local_output = tempfile.mkstemp(prefix='tmp-gameout')
        os.close(tempfd)
        clean_up_local_output = True

    try:
        combined_body = bytes()

        bucket, path = args.s3_input.split('/', 1)
        client = boto3.client('s3')
        keep_going = True
        continuation_token = None
        s3_args = {'Bucket': bucket, 'Prefix': path, 'MaxKeys': 100}
        while keep_going:
            response = client.list_objects_v2(**s3_args)
            for c in response['Contents']:
                obj_resp = client.get_object(
                    Bucket=bucket,
                    Key=c['Key'])
                combined_body += obj_resp['Body'].read()
            if response['IsTruncated']:
                s3_args['ContinuationToken'] = response['NextContinuationToken']
            else:
                keep_going = False

        with open(local_output, 'wb') as outf:
            outf.write(combined_body)

        if args.s3_output:
            bucket, key = args.s3_output.split('/', 1)
            s3 = boto3.resource('s3')
            s3.meta.client.upload_file(local_output, bucket, key)
    finally:
        if clean_up_local_output:
            os.unlink(local_output)


if __name__ == '__main__':
    main()
