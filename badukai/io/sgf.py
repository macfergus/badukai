from baduk import Player

__all__ = [
    'save_game_as_sgf',
]

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

def point_to_sgf(point, num_rows):
    row_idx = num_rows - point.row
    return ALPHABET[point.col - 1] + ALPHABET[row_idx]


def save_game_as_sgf(outf, game, game_result,
                     black_name=None,
                     white_name=None,
                     date=None,
                     komi=None):
    num_rows = game.board.num_rows
    outf.write('(;FF[4]')
    outf.write('GM[1]')
    outf.write('SZ[{}]'.format(num_rows))
    outf.write('RE[{}]'.format(str(game_result)))
    if black_name:
        outf.write('PB[{}]'.format(black_name))
    if white_name:
        outf.write('PW[{}]'.format(black_name))
    if date:
        outf.write('DT[{}]'.format(date.strftime('%Y-%m-%d')))
    if komi:
        outf.write('KM[{}]'.format(komi))
    moves = []
    while game.previous_state is not None:
        moves.append((game.previous_state.next_player, game.last_move))
        game = game.previous_state
    moves.reverse()
    for player, move in moves:
        player_name = 'B' if player == Player.black else 'W'
        if move.is_play:
            outf.write(';{player}[{point}]'.format(
                player=player_name,
                point=point_to_sgf(move.point, num_rows)))
        elif move.is_pass:
            outf.write(';{player}[]'.format(player=player_name))
    outf.write(')')
