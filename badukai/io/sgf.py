from collections import namedtuple

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from baduk import Board, GameState, Move, Player, Point

__all__ = [
    'GameRecord',
    'save_game_as_sgf',
    'read_game_from_sgf',
]

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

def point_to_sgf(point, num_rows):
    row_idx = num_rows - point.row
    return ALPHABET[point.col - 1] + ALPHABET[row_idx]


def decode_sgf_move(sgf_move, num_rows):
    if sgf_move == '':
        return Move.pass_turn()
    assert len(sgf_move) == 2
    col = ALPHABET.index(sgf_move[0]) + 1
    row = num_rows - ALPHABET.index(sgf_move[1])
    return Move.play(Point(row, col))


def save_game_as_sgf(outf, game, game_result,
                     handicap_stones=[],
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
        outf.write('PW[{}]'.format(white_name))
    if date:
        outf.write('DT[{}]'.format(date.strftime('%Y-%m-%d')))
    if komi:
        outf.write('KM[{}]'.format(komi))
    if handicap_stones:
        outf.write('HA[{}]'.format(len(handicap_stones)))
        outf.write('AB{}'.format(''.join(
            '[' + point_to_sgf(stone, num_rows) + ']'
            for stone in handicap_stones)))
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

sgf_grammar = Grammar('''
    collection = gametree+
    gametree = "(" sequence gametree* ")" whitespace
    sequence = node+
    node = ";" properties
    properties = padded_property*
    padded_property = property whitespace
    property = propident propvalues
    propident = ~"[A-Z][A-Z]?"
    propvalues = propvalue+
    propvalue = "[" value "]" whitespace
    value = ~"[^\]]*"
    whitespace = ~"\s*"
''')


class SGFProperty(namedtuple('SGFProperty', 'ident values')):
    pass


class SGFVisitor(NodeVisitor):
    """
    Only handles SGFs with a single sequence.
    """
    def visit_collection(self, node, children):
        # collection = gametree+
        gametrees = children[0]
        return gametrees

    def visit_gametree(self, node, children):
        # gametree = "(" sequence gametree* ")" whitespace
        _, sequence, _, _, _ = children
        return sequence

    def visit_sequence(self, node, children):
        # sequence = node+
        return children

    def visit_node(self, node, children):
        _, props = children
        return props

    def visit_padded_property(self, node, children):
        prop, _ = children
        return prop

    def visit_property(self, node, children):
        propident, propvalues = children
        return SGFProperty(propident, propvalues)

    def visit_propident(self, node, children):
        return node.text

    def visit_propvalue(self, node, children):
        _, value, _, _ = children
        return value

    def visit_propvalues(self, node, children):
        return children

    def visit_value(self, node, children):
        return node.text

    def visit_whitespace(self, node, whitespace):
        pass

    def generic_visit(self, node, children):
        return children


class GameRecord(namedtuple('GameRecord', 'initial_state moves winner')):
    pass


NO_DEFAULT = object()
def get_prop(proplist, ident, default=NO_DEFAULT):
    for prop in proplist:
        if prop.ident == ident:
            assert len(prop.values) == 1
            return prop.values[0]
    if default is NO_DEFAULT:
        raise KeyError(ident)
    return default


def get_props(proplist, ident):
    for prop in proplist:
        if prop.ident == ident:
            return prop.values
    raise KeyError(ident)


def read_game_from_sgf(sgffile):
    tree = sgf_grammar.parse(sgffile.read())
    visitor = SGFVisitor()
    sequence = visitor.visit(tree)
    root, moves = sequence[0], sequence[1:]
    board_size = int(get_prop(root, 'SZ'))
    try:
        komi = float(get_prop(root, 'KM'))
    except KeyError:
        # We don't truly support zero komi, pretend it's half point
        # komi.
        komi = 0.5
    result = get_prop(root, 'RE')
    if result.startswith('B+'):
        winner = Player.black
    elif result.startswith('W+'):
        winner = Player.white
    else:
        raise ValueError(result)
    handicap = int(get_prop(root, 'HA', 0))
    board = Board(board_size, board_size)
    first_player = Player.black
    if handicap > 0:
        handicap_stones = get_props(root, 'AB')
        for sgf_move in handicap_stones:
            move = decode_sgf_move(sgf_move, board_size)
            board.place_stone(Player.black, move.point)
            first_player = Player.white
    initial_state = GameState.from_board(board, first_player, komi)
    state = initial_state
    decoded_moves = []
    for node in moves:
        key = 'B' if state.next_player == Player.black else 'W'
        sgf_move = get_prop(node, key)
        move = decode_sgf_move(sgf_move, board_size)
        decoded_moves.append(move)
        state = state.apply_move(move)

    return GameRecord(
        initial_state=initial_state,
        moves=decoded_moves,
        winner=winner
    )
