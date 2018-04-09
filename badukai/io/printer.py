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

    def print_result(self, result):
        print(str(result))
