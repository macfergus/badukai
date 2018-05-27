import math
import random
from operator import attrgetter, itemgetter

import numpy as np
from keras.optimizers import Adadelta, SGD
from baduk import GameState

from ... import encoders
from ... import kerasutil
from ..base import Bot

__all__ = [
    'ZeroBot',
    'load_from_hdf5',
]


NOISE = 1e-4


COLS = 'ABCDEFGHJKLMNOPQRST'
def format_move(move):
    if move.is_pass:
        return 'pass'
    if move.is_resign:
        return 'resign'
    return COLS[move.point.col - 1] + str(move.point.row)


def print_tree(node, indent=''):
    for child_move, child_node in node._children.items():
        print("%s%s p %.6f vc %d q %.6f" % (
            indent, format_move(child_move),
            node._branches[child_move].prior,
            node._branches[child_move].visit_count,
            node._branches[child_move].total_value / node._branches[child_move].visit_count))
        print_tree(child_node, indent + '  ')


class Branch(object):
    def __init__(self, prior):
        # Prior probability of visiting this node from policy network
        self.prior = prior
        # Sum of estimated values over all visits
        self.total_value = 0.0
        # Number of rollouts that passed through this node
        self.visit_count = 0


class Node(object):
    def __init__(self, state, value, priors, last_move=None, parent=None):
        self.state = state
        self.value = value

        self.move = last_move
        self.parent = parent

        self.visit_count = 1
        self.total_value = self.value

        self.branches = {}
        for move in self.state.legal_moves():
            if not move.is_resign:
                prior = priors[move]
                self.branches[move] = Branch(prior)

        self.children = {}

        if parent:
            parent.add_child(last_move, self)

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return self.value
        return branch.total_value / branch.visit_count

    def uct_score(self, move):
        branch = self.branches[move]
        return branch.prior * (
            math.sqrt(self.visit_count) /
            (1 + branch.visit_count))

    def has_child(self, move):
        """Return True if the move has been added as a child."""
        return move in self.children

    def get_child(self, move):
        return self.children[move]

    def num_visits(self, move):
        if move in self.children:
            return self.children[move].visit_count
        return 0

    def get_children(self):
        # Make a copy
        return list(self.children.values())

    def record_visit(self, child_move, value):
        # The param value is from the point of view of the next state,
        # i.e., the opponent. So we invert.
        our_value = -1 * value
        self.visit_count += 1
        self.total_value += our_value
        self.branches[child_move].visit_count += 1
        self.branches[child_move].total_value += our_value
        if self.parent is not None:
            self.parent.record_visit(self.move, our_value)


class ZeroBot(Bot):
    def __init__(self, encoder, model):
        self._encoder = encoder
        self._model = model
        self._board_size = encoder.board_size()
        self._num_rollouts = 900
        self._temperature = 0.0
        self._exploration_factor = 2.0
        self.root = None

        self._noise_concentration = 0.03
        self._noise_weight = 0.25

    def name(self):
        return 'ZeroBot'

    def board_size(self):
        return self._board_size

    def set_option(self, name, value):
        if name == 'num_rollouts':
            self._num_rollouts = int(value)
        elif name == 'exploration':
            self._exploration_factor = float(value)
        else:
            raise KeyError(name)

    def set_temperature(self, temperature):
        self._temperature = temperature

    def select_move(self, game_state):
        self.root = self.create_node(game_state, add_noise=True)

        for i in range(self._num_rollouts):
            # Find a leaf.
            node = self.root
            path = []
            move = self.select_branch(node)
            path.append(move)
            while node.has_child(move):
                node = node.get_child(move)
                move = self.select_branch(node)
                path.append(move)

            # Expand the tree.
            new_child = self.create_child(node, move)

            # Update stats.
            node.record_visit(move, new_child.value)

        # Now select a move in proportion to how often we visited it.
        move_indices = []
        visit_counts = []
        for child in self.root.get_children():
            if child.visit_count > 0:
                move_indices.append(self._encoder.encode_move(child.move))
                visit_counts.append(child.visit_count)
        visit_counts = np.array(visit_counts)
        if self._temperature > 0:
            p = np.power(visit_counts, 1.0 / self._temperature)
            p /= np.sum(p)
            move_index = np.random.choice(move_indices, p=p)
        else:
            move_index = move_indices[np.argmax(visit_counts)]

        return self._encoder.decode_move_index(move_index)

    def create_child(self, parent, move):
        new_state = parent.state.apply_move(move)
        return self.create_node(new_state, parent, move)

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
        priors = {}
        for i, p in enumerate(probs):
            priors[self._encoder.decode_move_index(i)] = p

        new_node = Node(
            state,
            value,
            priors,
            last_move=move, parent=parent)
        return new_node

    def select_branch(self, node):
        scored_branches = []
        def branch_score(move):
            return node.expected_value(move) + \
                self._exploration_factor * node.uct_score(move)
        return max(node.branches.keys(), key=branch_score)

    def serialize(self, h5group):
        encoder_group = h5group.create_group('encoder')
        encoders.save_encoder(self._encoder, encoder_group)

        model_group = h5group.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, model_group)

    def search_counts(self):
        rv = []
        for idx in range(self._encoder.num_moves()):
            move = self._encoder.decode_move_index(idx)
            rv.append(self.root.num_visits(move))
        return rv

    def train(self, game_records, num_epochs=1):
        X = []
        y_policy = []
        y_value = []
        gn = 0
        for game_record in game_records:
            winner = game_record.winner
            print('Encoding game %d/%d...' % (gn + 1, len(game_records)))
            gn += 1
            game = GameState.new_game(self.board_size())
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
            SGD(lr=0.01, momentum=0.9),
            loss=['categorical_crossentropy', 'mse'])
        self._model.fit(
            X, [y_policy, y_value],
            batch_size=2048,
            epochs=num_epochs)

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


def load_from_hdf5(h5group):
    return ZeroBot(
        encoders.load_encoder(h5group['encoder']),
        kerasutil.load_model_from_hdf5_group(h5group['model'])
    )
