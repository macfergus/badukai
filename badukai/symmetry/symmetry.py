import baduk

__all__ = [
    'rotate_point',
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
