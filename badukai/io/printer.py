import baduk

__all__ = [
    'AsciiBoardPrinter',
]

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    baduk.Player.black: 'x',
    baduk.Player.white: 'o',
}


class AsciiBoardPrinter(object):
    def print_board(self, board):
        for row in range(board.num_rows, 0, -1):
            line = []
            for col in range(1, board.num_cols + 1):
                stone = board.get(baduk.Point(row=row, col=col))
                line.append(STONE_TO_CHAR[stone])
            print('%2d %s' % (row, ''.join(line)))
        print('   ' + COLS[:board.num_cols])

    def format_move(self, move):
        if move.is_resign:
            return 'resign'
        if move.is_pass:
            return 'pass'
        return '{}{}'.format(
            COLS[move.point.col - 1],
            move.point.row)

    def format_player(self, player):
        if player == baduk.Player.black:
            return 'B'
        if player == baduk.Player.white:
            return 'W'
        raise ValueError(player)

    def print_result(self, result):
        print(str(result))
