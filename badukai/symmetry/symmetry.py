import baduk

__all__ = [
    'rotate_point',
    'rotate_board',
]


def rotate_point(p, index, board_size):
    assert 0 <= index < 8
    if index & 1:
        # Reflect rows
        p = baduk.Point(row=board_size - p.row + 1, col=p.col)
    if index & 2:
        # Reflect cols
        p = baduk.Point(row=p.row, col=board_size - p.col + 1)
    if index & 4:
        # Swap rows & cols
        p = baduk.Point(row=p.col, col=p.row)
    return p


def rotate_board(board, index):
    size = board.num_rows
    assert board.num_cols == size
    new_board = baduk.Board(size, size)
    for r in range(1, size + 1):
        for c in range(1, size + 1):
            p = baduk.Point(row=r, col=c)
            stone = board.get(p)
            if stone is not None:
                new_point = rotate_point(p, index, size)
                new_board.place_stone(stone, new_point)
    return new_board
