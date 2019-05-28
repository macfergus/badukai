import math
import random
import sys
import time
from operator import attrgetter, itemgetter

import numpy as np
from keras.optimizers import Adadelta, SGD
from baduk import GameState, Move

from ... import encoders
from ... import kerasutil
from ... import schedules
from ..base import Bot

__all__ = [
    'ZeroBot',
    'load_from_hdf5',
]


NOISE = 1e-4
VIRTUAL_LOSS_FRAC = 0.8


def get_ladder_points(state):
    legal_moves = state.legal_moves_as_array()
    capture_points = state.board.liberties_as_array(
        state.next_player.other, 1
    ).flatten()
    capture_points.resize(legal_moves.shape)
    capture_points *= legal_moves
    if np.sum(capture_points) > 0:
        return capture_points
    escape_points = state.board.liberties_as_array(
        state.next_player, 1
    ).flatten()
    escape_points.resize(legal_moves.shape)
    escape_points *= legal_moves
    if np.sum(escape_points) > 0:
        return escape_points
    ladder_points = state.board.liberties_as_array(
        state.next_player.other, 2
    ).flatten()
    ladder_points = ladder_points.flatten()
    ladder_points.resize(legal_moves.shape)
    ladder_points *= legal_moves
    return ladder_points


def calc_expected_values(total_values, visit_counts, virtual_losses=None, vl_value=0.0):
    if virtual_losses is None:
        virtual_losses = np.zeros_like(total_values)
    virtual_visit_counts = visit_counts + virtual_losses
    return np.divide(
        total_values + vl_value * virtual_losses,
        virtual_visit_counts,
        out=-1 * np.ones_like(total_values),
        where=virtual_visit_counts > 0)


COLS = 'ABCDEFGHJKLMNOPQRST'
def format_move(move):
    if move.is_pass:
        return 'pass'
    if move.is_resign:
        return 'resign'
    return COLS[move.point.col - 1] + str(move.point.row)


def print_tree(node, encoder, indent=''):
    expected_values = np.divide(
        node.total_values,
        node.visit_counts,
        out=-1 * np.ones_like(node.total_values),
        where=node.visit_counts > 0)
    v_visits = node.visit_counts + node.virtual_losses
    v_expected_values = np.divide(
        node.total_values + node.value * node.virtual_losses,
        v_visits,
        out=-1 * np.ones_like(node.total_values),
        where=v_visits > 0)
    for child_move_idx, child_node in enumerate(node.children):
        if v_visits[child_move_idx] == 0:
            continue
        print("%s%s p %.6f vc %d vl %.3f q %.6f vq %.6f" % (
            indent, format_move(encoder.decode_move_index(child_move_idx)),
            node.priors[child_move_idx],
            node.visit_counts[child_move_idx],
            node.virtual_losses[child_move_idx],
            expected_values[child_move_idx],
            v_expected_values[child_move_idx]))
        if child_node is not None:
            print_tree(child_node, encoder, indent + '  ')


class Node(object):
    def __init__(self, state, value, priors, legal_moves,
                 last_move=None, parent=None):
        self.priors = priors
        self.visit_counts = np.zeros_like(priors)
        self.virtual_losses = np.zeros_like(priors)
        self.total_values = np.zeros_like(priors)
        self.is_legal = legal_moves
        self.children = [None for _ in priors]

        self.state = state
        self.value = value

        self.move = last_move
        self.parent = parent

        if parent:
            parent.add_child(last_move, self)

    def __hash__(self):
        return hash(id(self)) # HACK

    def __eq__(self, other):
        return self is other # HACK

    def add_virtual_loss(self, move):
        self.virtual_losses[move] += 1

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        """Return True if the move has been added as a child."""
        return self.children[move] is not None

    def get_child(self, move):
        return self.children[move]

    def record_visit(self, child_move, value):
        # The param value is from the point of view of the next state,
        # i.e., the opponent. So we invert.
        our_value = -1 * value
        self.visit_counts[child_move] += 1
        self.total_values[child_move] += our_value
        if self.parent is not None:
            self.parent.record_visit(self.move, our_value)
        # Reset virtual losses after recording a real visit.
        self.virtual_losses[:] = 0


class TacNode:
    """Tree node for tactical reading.

    This just reads out atari-extend sequences in a minimax fashion.
    """
    def __init__(self, state, value, ladder_points, priors, legal_moves):
        self.state = state
        self.value = value
        self.ladder_points = ladder_points
        self.priors = priors
        self.legal_moves = legal_moves
        self.children = {}

    @property
    def n_children(self):
        return np.sum(self.ladder_points)


def get_minimax_value(tac_node):
    if not tac_node.children:
        return tac_node.value
    best_value = -2
    for _, child in tac_node.children.items():
        child_value = -1 * get_minimax_value(child)
        if child_value > best_value:
            best_value = child_value
    return best_value


def convert_tree(tac_node, last_move=None, parent=None):
    value = get_minimax_value(tac_node)
    node = Node(
        tac_node.state,
        tac_node.value,
        tac_node.priors,
        tac_node.legal_moves,
        last_move,
        parent
    )
    if not tac_node.children:
        return
    best_value = -2
    best_move = -1
    for move_idx, child in tac_node.children.items():
        child_value = -1 * get_minimax_value(child)
        if child_value > best_value:
            best_value = child_value
            best_move = move_idx
    child = tac_node.children[best_move]
    child_node = convert_tree(
        child,
        best_move,
        node
    )
    node.record_visit(best_move, -1 * best_value)
    return node


class ZeroBot(Bot):
    def __init__(self, encoder, model):
        self._encoder = encoder
        self._model = model
        self._board_size = encoder.board_size()
        self._num_rollouts = 900
        self._ladder_rollouts = 0
        self._exploration_factor = 1.25
        self._batch_size = 8
        self._resign_below = -2.0
        self._temp_schedule = schedules.ConstantSchedule(0.0)
        self._debug_log = True
        self.root = None

        self._noise_concentration = 0.03
        self._noise_weight = 0.25

        self._gracious_winner = None

    def name(self):
        return 'ZeroBot'

    def board_size(self):
        return self._board_size

    def set_option(self, name, value):
        if name == 'num_rollouts':
            self._num_rollouts = int(value)
        elif name == 'ladder_rollouts':
            self._ladder_rollouts = int(value)
        elif name == 'exploration':
            self._exploration_factor = float(value)
        elif name == 'batch_size':
            self._batch_size = int(value)
        elif name == 'temperature':
            self._temp_schedule = schedules.parse_schedule(value)
        elif name == 'resign_below':
            self._resign_below = float(value)
        elif name == 'noise_weight':
            self._noise_weight = float(value)
        elif name == 'gracious_winner':
            self._gracious_winner = float(value)
        else:
            raise KeyError(name)

    def set_temperature(self, temperature):
        self._temperature = temperature

    def read_ladders(self, game_state, max_rollouts):
        start = time.time()
        from ...io import AsciiBoardPrinter
        pp = AsciiBoardPrinter()
        root = self.create_tac_root(game_state)
        if root.n_children == 0:
            return None
        num_rollouts = 0
        while num_rollouts < max_rollouts:
            # Walk down the minimax-optimal path.
            node, value = self.get_minimax_best(root)
            if node is None:
                break
            # Get ladder points from this state.
            # Add children to the tree for each ladder point.
            move_idxs, = node.ladder_points.nonzero()
            pairs = []
            for move_idx in move_idxs:
                pairs.append((
                    move_idx,
                    node.state.apply_move(
                        self._encoder.decode_move_index(move_idx)
                    )
                ))
            assert node.n_children > 0
            assert node.n_children == len(pairs)
            self.create_tac_nodes(node, pairs)
            num_rollouts += 1
        end = time.time()
        elapsed = end - start
        sys.stderr.write(
            'Ladder check: {} rollouts in {:.3f} seconds\n'.format(
                num_rollouts, elapsed
            )
        )
        #pp.print_board(game_state.board)
        #for move_idx, child in root.children.items():
        #    move = self._encoder.decode_move_index(move_idx)
        #    leaf, leaf_value = self.get_minimax_leaf(child)
        #    print(' {}: {:.3f}'.format(
        #        format_move(move),
        #        -1 * leaf_value
        #    ))
        #    pp.print_board(leaf.state.board)
        #    print(leaf.value, leaf.state.next_player)

        new_root = convert_tree(root)
        #print_tree(new_root, self._encoder)
        return new_root

    def get_minimax_best(self, node):
        if not node.children:
            return node, node.value
        best_leaf = None
        best_value = -2
        for _, child in node.children.items():
            leaf, value = self.get_minimax_best(child)
            if leaf is None:
                continue
            if leaf.n_children > 0:
                my_value = -1 * value
                if best_leaf is None or best_value < my_value:
                    best_leaf = leaf
                    best_value = my_value
        return best_leaf, best_value

    def get_minimax_leaf(self, node):
        if not node.children:
            return node, node.value
        best_value = -2
        best_node = None
        for _, child in node.children.items():
            leaf, value = self.get_minimax_leaf(child)
            my_value = -1 * value
            if my_value > best_value:
                best_value = my_value
                best_node = leaf
        return best_node, best_value

    def create_tac_root(self, game_state):
        tensors = [self._encoder.encode(game_state)]
        tensors = np.array(tensors)
        priors, values = self._model.predict(tensors)
        legal_moves = game_state.legal_moves_as_array()
        ladder_points = get_ladder_points(game_state)
        return TacNode(
            game_state,
            values[0][0],
            ladder_points,
            priors[0],
            legal_moves
        )

    def create_tac_nodes(self, node, pairs):
        tensors = [self._encoder.encode(state) for _, state in pairs]
        tensors = np.array(tensors)
        probs, values = self._model.predict(tensors)
        for i, (move_index, next_state) in enumerate(pairs):
            value = values[i][0]
            legal_moves = next_state.legal_moves_as_array()
            ladder_points = get_ladder_points(next_state)
            move_idxs, = ladder_points.nonzero()
            for move_idx in move_idxs:
                move = self._encoder.decode_move_index(move_idx)
                assert move.is_play
                assert next_state.board.is_empty(move.point)
            if node is not None:
                node.children[move_index] = TacNode(
                    next_state,
                    value,
                    ladder_points,
                    probs[i],
                    legal_moves
                )

    def select_move(self, game_state):
        start = time.time()
        self.root = None
        if self._ladder_rollouts > 0:
            self.root = self.read_ladders(game_state, self._ladder_rollouts)
        if self.root is None:
            self.root = self.create_node(game_state, add_noise=True)

        num_rollouts = 0
        while num_rollouts < self._num_rollouts:
            to_expand = set()
            batch_count = 0
            while batch_count < self._batch_size:
                # Find a leaf.
                node = self.root
                move = self.select_branch(node)
                while node.has_child(move):
                    node.add_virtual_loss(move)
                    node = node.get_child(move)
                    move = self.select_branch(node)
                node.add_virtual_loss(move)
                batch_count += 1
                to_expand.add((node, move))

            batch_num_visits = len(to_expand)
            new_children = self.create_children(to_expand)
            for new_child in new_children:
                new_child.parent.record_visit(
                    new_child.move, new_child.value)
            num_rollouts += batch_num_visits

        # Now select a move in proportion to how often we visited it.
        visit_counts = self.root.visit_counts
        expected_values = calc_expected_values(self.root.total_values, visit_counts)
        tiebreak = 0.499 * (expected_values + 1)
        decide_vals = visit_counts + tiebreak
        for move_idx in np.argsort(decide_vals):
            visit_count = visit_counts[move_idx]
            if visit_count > 0:
                sys.stderr.write('{}: {:.3f} {}\n'.format(
                    format_move(self._encoder.decode_move_index(move_idx)),
                    expected_values[move_idx],
                    visit_count))
        temperature = self._temp_schedule.get(game_state.num_moves)
        if temperature > 0:
            move_indices, = np.where(visit_counts > 0)
            raw_counts = decide_vals[move_indices]
            p = np.power(raw_counts, 1.0 / temperature)
            p /= np.sum(p)
            move_index = np.random.choice(move_indices, p=p)
        else:
            move_index = np.argmax(decide_vals)

        self._log_pv(self.root)

        chosen_move = self._encoder.decode_move_index(move_index)
        sys.stderr.write('Select {} Q {:.3f}\n'.format(
            format_move(chosen_move),
            expected_values[move_index]))
        end = time.time()
        sys.stderr.write('Decided in {:.3f}s\n'.format(end - start))
        sys.stderr.flush()
        if expected_values[move_index] < self._resign_below:
            sys.stderr.write('Resigning because Q {:.3f} < {:.3f}\n'.format(
                expected_values[move_index],
                self._resign_below))
            return Move.resign()

        if self._gracious_winner is not None:
            if game_state.last_move is not None and game_state.last_move == Move.pass_turn():
                pass_idx = self._encoder.encode_move(Move.pass_turn())
                if visit_counts[pass_idx] >= 2 and \
                        expected_values[pass_idx] > self._gracious_winner:
                    sys.stderr.write('Pass has Q {:.3f}\n'.format(expected_values[pass_idx]))
                    return Move.pass_turn()
        return chosen_move

    def evaluate(self, game_state):
        state_tensor = self._encoder.encode(game_state)
        _, values = self._model.predict(np.array([state_tensor]))
        return values[0][0]

    def create_children(self, pairs):
        states = []
        state_tensors = []
        for parent, move_idx in pairs:
            move = self._encoder.decode_move_index(move_idx)
            next_state = parent.state.apply_move(move)
            states.append(next_state)
            state_tensors.append(self._encoder.encode(next_state))
        state_tensors = np.array(state_tensors)

        probs, values = self._model.predict(state_tensors)

        new_nodes = []
        for i, (parent, move_idx) in enumerate(pairs):
            move_probs = probs[i]
            value = values[i][0]

            new_node = Node(
                states[i],
                value,
                priors=move_probs,
                legal_moves=self._legal_move_mask(states[i]),
                last_move=move_idx, parent=parent)
            new_nodes.append(new_node)
        return new_nodes

    def _legal_move_mask(self, state):
        # WARNING! This breaks the encode_move abstraction!
        return state.legal_moves_as_array()
        legal_moves = np.zeros(self._encoder.num_moves())
        for move in state.legal_moves():
            if not move.is_resign:
                legal_moves[self._encoder.encode_move(move)] = 1
        return legal_moves

    def create_node(self, state, parent=None, move=None, add_noise=False):
        state_tensor = self._encoder.encode(state)
        state_tensors = np.array([state_tensor])

        probs, values = self._model.predict(state_tensors)

        probs = probs[0]
        if add_noise:
            noise = np.random.dirichlet(
                self._noise_concentration * np.ones(self._encoder.num_moves()))
            probs = (1 - self._noise_weight) * probs + \
                self._noise_weight * noise

        value = values[0][0]

        move_idx = None if move is None \
            else self._encoder.encode_move(move)

        new_node = Node(
            state,
            value,
            priors=probs,
            legal_moves=self._legal_move_mask(state),
            last_move=move_idx,
            parent=parent)
        return new_node

    def select_branch(self, node):
        visits = node.visit_counts + node.virtual_losses
        vl_value = VIRTUAL_LOSS_FRAC * (node.value - -1) + (-1)
        expected_values = calc_expected_values(
            node.total_values,
            node.visit_counts,
            virtual_losses=node.virtual_losses,
            vl_value=vl_value)

        total_visits = 1 + np.sum(visits)
        uct_scores = node.priors * np.sqrt(total_visits) / (1 + visits)

        branch_scores = (
            expected_values + self._exploration_factor * uct_scores)

        # Avoid selecting illegal moves.
        branch_scores[node.is_legal == 0] = np.min(branch_scores) - 1
        to_select = np.argmax(branch_scores)
        assert node.is_legal[to_select] > 0
        return to_select

    def serialize(self, h5group):
        encoder_group = h5group.create_group('encoder')
        encoders.save_encoder(self._encoder, encoder_group)

        model_group = h5group.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, model_group)

    def search_counts(self):
        return [int(x) for x in self.root.visit_counts]

    def train(self, game_records, num_epochs=1):
        X = []
        y_policy = []
        y_value = []
        gn = 0
        for game_record in game_records:
            winner = game_record.winner
            print('Encoding game %d/%d...' % (gn + 1, len(game_records)))
            gn += 1
            game = game_record.initial_state
            for move_record in game_record.move_records:
                assert move_record.player == game.next_player
                X.append(self._encoder.encode(game))
                search_counts = np.array(move_record.visit_counts)
                y_policy.append(search_counts / np.sum(search_counts))
                y_value.append(1 if game.next_player == winner else -1)
                game = game.apply_move(move_record.move)
        X = np.array(X)
        y_policy = np.array(y_policy)
        y_value = np.array(y_value)
        print(X.shape, y_policy.shape, y_value.shape)

        self._model.compile(
            SGD(lr=0.002, momentum=0.9),
            loss=['categorical_crossentropy', 'mse'],
            loss_weights=[1.0, 0.1])
        self._model.fit(
            X, [y_policy, y_value],
            batch_size=512,
            epochs=num_epochs)

    def train_direct(self, X, y_policy, y_value,
                     epochs=1,
                     optimizer='sgd',
                     lr=0.05, momentum=0.9,
                     batch_size=512,
                     value_weight=0.1,
                     validation_frac=0.0):
        if optimizer == 'sgd':
            opt = SGD(lr=lr, momentum=momentum)
        elif optimizer == 'adadelta':
            opt = Adadelta()
        self._model.compile(
            opt,
            loss=['categorical_crossentropy', 'mse'],
            loss_weights=[1.0, value_weight])
        self._model.fit(
            X, [y_policy, y_value],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_frac)

    def encode_game(self, game_record):
        X = []
        y_policy = []
        y_value = []
        winner = game_record.winner
        game = game_record.initial_state
        for move_record in game_record.move_records:
            assert move_record.player == game.next_player
            X.append(self._encoder.encode(game))
            search_counts = np.array(move_record.visit_counts)
            y_policy.append(search_counts / np.sum(search_counts))
            y_value.append(1 if game.next_player == winner else -1)
            game = game.apply_move(move_record.move)
        X = np.array(X)
        y_policy = np.array(y_policy)
        y_value = np.array(y_value)
        return X, y_policy, y_value

    def train_from_human(self, human_game_records, batch_size=2048):
        X = []
        y_policy = []
        y_value = []
        gn = 0
        for game_record in human_game_records:
            print('Encoding game %d/%d...' % (gn + 1, len(human_game_records)))
            gn += 1
            winner = game_record.winner
            game = game_record.initial_state
            for move in game_record.moves:
                X.append(self._encoder.encode(game))
                search_counts = np.zeros(self._encoder.num_moves())
                search_counts[self._encoder.encode_move(move)] = 1
                y_policy.append(search_counts)
                y_value.append(1 if game.next_player == winner else -1)
                game = game.apply_move(move)
        X = np.array(X)
        y_policy = np.array(y_policy)
        y_value = np.array(y_value)
        print(X.shape, y_policy.shape, y_value.shape)

        self._model.compile(
            Adadelta(),
            loss=['categorical_crossentropy', 'mse'],
            # Seeing massive overfitting on individual game results.
            # Hopefully lowering the loss weight helps.
            loss_weights=[1.0, 0.1])
        self._model.fit(
            X, [y_policy, y_value],
            batch_size=batch_size,
            epochs=1)

    def encode_human(self, human_game_records, discount_rate=0.995):
        X = []
        y_policy = []
        y_value = []
        gn = 0
        for game_record in human_game_records:
            gn += 1
            winner = game_record.winner
            game = game_record.initial_state
            values = []
            for move in game_record.moves:
                X.append(self._encoder.encode(game))
                search_counts = np.zeros(self._encoder.num_moves())
                search_counts[self._encoder.encode_move(move)] = 1
                y_policy.append(search_counts)
                values.append(1 if game.next_player == winner else -1)
                game = game.apply_move(move)
            num_moves = len(values)
            for i in range(num_moves):
                distance = num_moves - (i + 1)
                values[i] *= discount_rate ** distance
            y_value += values
        X = np.array(X)
        y_policy = np.array(y_policy)
        y_value = np.array(y_value)
        print(X.shape, y_policy.shape, y_value.shape)
        return X, y_policy, y_value

    def _get_pv(self, node):
        best_idx = np.argmax(node.visit_counts)
        if node.visit_counts[best_idx] < 2:
            return []
        return [
            (
                self._encoder.decode_move_index(best_idx),
                node.visit_counts[best_idx]
            )] + self._get_pv(node.children[best_idx])

    def _log_pv(self, node):
        if self._debug_log:
            pv = self._get_pv(node)
            sys.stderr.write('PV: {}\n'.format(
                ' / '.join('{} {}'.format(format_move(m), c) for m,c in pv)
            ))


def load_from_hdf5(h5group):
    return ZeroBot(
        encoders.load_encoder(h5group['encoder']),
        kerasutil.load_model_from_hdf5_group(h5group['model'])
    )
