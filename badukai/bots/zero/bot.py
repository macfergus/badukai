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
    def __init__(self, encoder, model, state, move=None, parent=None):
        self.move = move
        self.parent = parent
        self._state = state
        self._exploration_factor = 2.0

        self._encoder = encoder
        self._model = model

        # Feed the model to get
        # a) the initial expected value of this state
        # b) the priors for the child branches

        state_tensor = np.array([encoder.encode(state)])
        probs, values = model.predict(state_tensor)
        probs = probs[0]
        self.value = values[0][0]
        self.visit_count = 1
        self.total_value = self.value

        self._branches = {}
        for move in state.legal_moves():
            if not move.is_resign:
                prior = probs[encoder.encode_move(move)]
                self._branches[move] = Branch(prior)

        self._children = {}

    def _expected_value(self, move):
        branch = self._branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def _uct_score(self, move):
        branch = self._branches[move]
        return branch.prior * (
            math.sqrt(self.visit_count) /
            (1 + branch.visit_count))

    def select_branch(self):
        scored_branches = []
        for move in self._branches:
            score = self._expected_value(move) + \
                self._exploration_factor * self._uct_score(move)
            scored_branches.append((score, move))
        scored_branches.sort(key=itemgetter(0))
        return scored_branches[-1][-1]

    def has_child(self, move):
        """Return True if the move has been added as a child."""
        return move in self._children

    def get_child(self, move):
        return self._children[move]

    def num_visits(self, move):
        if move in self._children:
            return self._children[move].visit_count
        return 0

    def get_children(self):
        # Make a copy
        return list(self._children.values())

    def create_child(self, move):
        """Add the move as a child in the tree."""
        next_state = self._state.apply_move(move)

        child = Node(
            self._encoder, self._model,
            next_state,
            move=move, parent=self)
        self._children[move] = child
        return child

    def record_visit(self, child_move, value):
        # The param value is from the point of view of the next state,
        # i.e., the opponent. So we invert.
        our_value = -1 * value
        self.visit_count += 1
        self.total_value += our_value
        self._branches[child_move].visit_count += 1
        self._branches[child_move].total_value += our_value
        if self.parent is not None:
            self.parent.record_visit(self.move, our_value)

    def add_noise(self, noise, weight):
        for i, noise_amount in enumerate(noise):
            move = self._encoder.decode_move_index(i)
            if move in self._branches:
                self._branches[move].prior = \
                    (1 - weight) * self._branches[move].prior + \
                    weight * noise_amount


class ZeroBot(Bot):
    def __init__(self, encoder, model):
        self._encoder = encoder
        self._model = model
        self._board_size = encoder.board_size()
        self._num_rollouts = 900
        self._temperature = 0.0
        self.root = None

        self._noise_concentration = 0.03
        self._noise_weight = 0.25

    def name(self):
        return 'ZeroBot'

    def board_size(self):
        return self._board_size

    def set_num_rollouts(self, num_rollouts):
        self._num_rollouts = num_rollouts

    def set_temperature(self, temperature):
        self._temperature = temperature

    def select_move(self, game_state, temperature=0):
        self.root = Node(self._encoder, self._model, game_state)
        self.root.add_noise(
            np.random.dirichlet(
                self._noise_concentration * np.ones(self._encoder.num_moves())),
            self._noise_weight)

        for i in range(self._num_rollouts):
            # Find a leaf.
            node = self.root
            path = []
            move = node.select_branch()
            path.append(move)
            while node.has_child(move):
                node = node.get_child(move)
                move = node.select_branch()
                path.append(move)

            # Expand the tree.
            new_child = node.create_child(move)

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
