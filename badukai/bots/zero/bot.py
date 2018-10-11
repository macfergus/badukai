import math
import random
import sys
from operator import attrgetter, itemgetter

import numpy as np
from keras.optimizers import Adadelta, SGD
from baduk import GameState, Move

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


class ZeroBot(Bot):
    def __init__(self, encoder, model):
        self._encoder = encoder
        self._model = model
        self._board_size = encoder.board_size()
        self._num_rollouts = 900
        self._temperature = 0.0
        self._exploration_factor = 1.25
        self._batch_size = 8
        self._resign_below = -2.0
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
        elif name == 'batch_size':
            self._batch_size = int(value)
        elif name == 'temperature':
            self._temperature = float(value)
        elif name == 'resign_below':
            self._resign_below = float(value)
        else:
            raise KeyError(name)

    def set_temperature(self, temperature):
        self._temperature = temperature

    def select_move(self, game_state):
        self.root = self.create_node(game_state, add_noise=True)

        num_rollouts = 0
        while num_rollouts < self._num_rollouts:
            to_expand = set()
            while len(to_expand) < self._batch_size:
                # Find a leaf.
                node = self.root
                move = self.select_branch(node)
                while node.has_child(move):
                    node.add_virtual_loss(move)
                    node = node.get_child(move)
                    move = self.select_branch(node)
                node.add_virtual_loss(move)
                to_expand.add((node, move))

            new_children = self.create_children(to_expand)
            for new_child in new_children:
                new_child.parent.record_visit(
                    new_child.move, new_child.value)
            num_rollouts += self._batch_size

        # Now select a move in proportion to how often we visited it.
        visit_counts = self.root.visit_counts
        expected_values = np.divide(
            self.root.total_values,
            visit_counts,
            out=np.zeros_like(self.root.total_values),
            where=visit_counts > 0)
        for move_idx in np.argsort(visit_counts):
            visit_count = visit_counts[move_idx]
            if visit_count > 0:
                sys.stderr.write('{}: {:.3f} {}\n'.format(
                    format_move(self._encoder.decode_move_index(move_idx)),
                    expected_values[move_idx],
                    visit_count))
        if self._temperature > 0:
            move_indices, = np.where(visit_counts > 0)
            raw_counts = visit_counts[move_indices]
            p = np.power(raw_counts, 1.0 / self._temperature)
            p /= np.sum(p)
            move_index = np.random.choice(move_indices, p=p)
        else:
            move_index = np.argmax(visit_counts)

        chosen_move = self._encoder.decode_move_index(move_index)
        sys.stderr.write('Select {} Q {:.3f}\n'.format(
            format_move(chosen_move),
            expected_values[move_index]))
        sys.stderr.flush()
        if expected_values[move_index] < self._resign_below:
            sys.stderr.write('Resigning because Q {:.3f} < {:.3f}\n'.format(
                expected_values[move_index],
                self._resign_below))
            return Move.resign()
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
        expected_values = np.divide(
            node.total_values,
            visits,
            out=np.zeros_like(node.total_values),
            where=visits > 0)
        
        total_visits = np.sum(node.visit_counts) + np.sum(node.virtual_losses)
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
        return self.root.visit_counts

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
