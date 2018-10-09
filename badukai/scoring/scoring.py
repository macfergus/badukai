from collections import namedtuple

from baduk import GameState, Move, Player, Point, remove_dead_stones


__all__ = [
    'GameResult',
    'compute_game_result',
    'remove_dead_stones_and_score',
]


class Territory(object):
    def __init__(self, territory_map):
        self.num_black_territory = 0
        self.num_white_territory = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_dame = 0
        self.dame_points = []
        for point, status in territory_map.items():
            if status == Player.black:
                self.num_black_stones += 1
            elif status == Player.white:
                self.num_white_stones += 1
            elif status == 'territory_b':
                self.num_black_territory += 1
            elif status == 'territory_w':
                self.num_white_territory += 1
            elif status == 'dame':
                self.num_dame += 1
                self.dame_points.append(point)


class GameResult(namedtuple('GameResult',
                            'b_resign w_resign b w komi final_board')):
    @property
    def winner(self):
        if self.b_resign:
            return Player.white
        if self.w_resign:
            return Player.black
        if self.b > self.w + self.komi:
            return Player.black
        return Player.white

    @property
    def winning_margin(self):
        w = self.w + self.komi
        return abs(self.b - w)

    def __str__(self):
        if self.b_resign:
            return 'W+R'
        if self.w_resign:
            return 'B+R'
        w = self.w + self.komi
        if self.b > w:
            return 'B+%.1f' % (self.b - w,)
        return 'W+%.1f' % (w - self.b,)


""" evaluate_territory:
Map a board into territory and dame.

Any points that are completely surrounded by a single color are
counted as territory; it makes no attempt to identify even
trivially dead groups.
"""


# tag::scoring_evaluate_territory[]
def evaluate_territory(board):

    status = {}
    for r in range(1, board.num_rows + 1):
        for c in range(1, board.num_cols + 1):
            p = Point(row=r, col=c)
            if p in status:  # <1>
                continue
            stone = board.get(p)
            if stone is not None:  # <2>
                status[p] = board.get(p)
            else:
                group, neighbors = _collect_region(p, board)
                if len(neighbors) == 1:  # <3>
                    neighbor_stone = neighbors.pop()
                    stone_str = 'b' if neighbor_stone == Player.black else 'w'
                    fill_with = 'territory_' + stone_str
                else:
                    fill_with = 'dame'  # <4>
                for pos in group:
                    status[pos] = fill_with
    return Territory(status)


def _collect_region(start_pos, board, visited=None):
    if visited is None:
        visited = {}
    if start_pos in visited:
        return [], set()
    all_points = [start_pos]
    all_borders = set()
    visited[start_pos] = True
    here = board.get(start_pos)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for delta_r, delta_c in deltas:
        next_p = Point(row=start_pos.row + delta_r, col=start_pos.col + delta_c)
        if not board.is_on_grid(next_p):
            continue
        neighbor = board.get(next_p)
        if neighbor == here:
            points, borders = _collect_region(next_p, board, visited)
            all_points += points
            all_borders |= borders
        else:
            all_borders.add(neighbor)
    return all_points, all_borders


def compute_game_result(game_state):
    b_resign = False
    w_resign = False
    if game_state.last_move.is_resign:
        if game_state.next_player == Player.black:
            w_resign = True
        else:
            b_resign = True

    territory = evaluate_territory(game_state.board)
    return GameResult(
        b_resign,
        w_resign,
        territory.num_black_territory + territory.num_black_stones,
        territory.num_white_territory + territory.num_white_stones,
        komi=game_state.komi(),
        final_board=game_state.board)


def remove_dead_stones_and_score(game_state):
    b_resign = False
    w_resign = False
    if game_state.last_move.is_resign:
        if game_state.next_player == Player.black:
            w_resign = True
        else:
            b_resign = True
    
    if b_resign or w_resign:
        return GameResult(
            b_resign, w_resign, 
            0.0, 0.0,
            game_state.komi(), game_state.board)

    end_game = game_state
    while end_game.last_move == Move.pass_turn():
        end_game = end_game.previous_state
    final_board = remove_dead_stones(end_game)
    final_state = GameState.from_board(
        final_board,
        game_state.next_player,
        game_state.komi())
    final_state = final_state.apply_move(Move.pass_turn())
    final_state = final_state.apply_move(Move.pass_turn())
    return compute_game_result(final_state)
