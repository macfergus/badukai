import argparse
import collections
import json
import os

import baduk

import badukai


def decode_ogs_move(omove):
    x, y, t = omove
    if x < 0:
        move = baduk.Move.pass_turn()
    else:
        col = x + 1
        row = 19 - y
        move = baduk.Move.play(baduk.Point(row, col))
    return move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auth', type=str, required=True)
    parser.add_argument('--secret', type=str, required=True)
    parser.add_argument('--num-games', '-n', type=int, default=100)
    parser.add_argument('--start-page', type=int)
    parser.add_argument('--output', '-o', required=True)
    args = parser.parse_args()

    with badukai.io.get_input(args.secret) as secret_file:
        secrets = json.load(open(secret_file))

    client = badukai.ogs.get_client(
        'https://online-go.com',
        args.auth, secrets)

    user = client.make_request('/api/v1/me')
    user_id = user['id']
    username = user['username']
    print('Loading games for {}...'.format(username))

    opponents = collections.Counter()
    total = 0
    for game in badukai.ogs.get_game_records(client, start_page=args.start_page):
        game_id = game['id']
        i_was_black = game['players']['black']['id'] == user_id
        i_was_white = game['players']['white']['id'] == user_id
        detail_path = game['related']['detail']
        fname = '{}.sgf'.format(game['id'])
        output_fname = os.path.join(args.output, fname)
        if os.path.exists(output_fname):
            print('Already downloaded {}'.format(fname))
            continue
        loss = False
        opponent = None
        if not game.get('outcome'):
            print('Game is not over')
            continue
        if i_was_black and game['black_lost']:
            opponent = game['players']['white']['username']
            loss = True
        if i_was_white and game['white_lost']:
            opponent = game['players']['black']['username']
            loss = True
        if loss:
            details = client.make_request(detail_path)
            if 'gamedata' not in details:
                print('What up with these details?')
                print(details)
                continue
            komi = details['gamedata']['komi']
            if len(details['gamedata']['moves']) < 30:
                print('Too short.')
            elif details['gamedata']['start_time'] < 1540364400:
                # was 1527577200
                print('Too old.')
            elif komi < -9 or komi > 9:
                print('Too much komi.')
            else:
                opponents[opponent] += 1
                total += 1
                handicap = details['gamedata']['handicap']
                handicap_stones = []
                moves = details['gamedata']['moves']
                board = baduk.Board(19, 19)
                handicap_setup = moves[:handicap]
                game_moves = moves[handicap:]
                if handicap > 0:
                    for setup in handicap_setup:
                        move = decode_ogs_move(setup)
                        assert(move.is_play)
                        handicap_stones.append(move.point)
                        board.place_stone(baduk.Player.black, move.point)
                    player = baduk.Player.white
                else:
                    player = baduk.Player.black
                g = baduk.GameState.from_board(board, player, komi)
                for m in game_moves:
                    move = decode_ogs_move(m)
                    g = g.apply_move(move)
                print('Saving {}'.format(fname))
                with open(output_fname, 'w') as outf:
                    badukai.io.save_game_as_sgf(
                        outf,
                        g,
                        'B+?' if i_was_white else 'W+?',
                        komi=komi,
                        handicap_stones=handicap_stones)
    print('Total: {} losses'.format(total))
    print(opponents)

if __name__ == '__main__':
    main()
