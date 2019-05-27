import argparse

from keras import regularizers
from keras.layers import Activation, BatchNormalization, Conv2D, Dense
from keras.layers import Flatten, Input
from keras.models import Model

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=19)
    parser.add_argument('--encoder', '-e')
    parser.add_argument('--network', '-n')
    parser.add_argument('botfile')
    args = parser.parse_args()

    encoder = badukai.encoders.get_encoder_by_name(
        args.encoder, args.board_size)

    # Build the policy-value network.
    # First the conv layers, which are then split into two outputs
    board_input = Input(shape=encoder.shape(), name='board_input')
    processed_board = board_input
    layers = badukai.networks.get_network_by_name(
        args.network, encoder.shape())
    for layer in layers:
        processed_board = layer(processed_board)

    # Policy output
    policy_conv = Conv2D(
        2, (1, 1), data_format='channels_first',
        kernel_regularizer=regularizers.l2(0.01))(processed_board)
    policy_batch = BatchNormalization(axis=1)(policy_conv)
    policy_relu = Activation('relu')(policy_batch)
    policy_flat = Flatten()(policy_relu)
    policy_output = Dense(
        encoder.num_moves(), activation='softmax',
        kernel_regularizer=regularizers.l2(0.01))(policy_flat)

    # Value output
    value_conv = Conv2D(
        1, (1, 1), data_format='channels_first',
        kernel_regularizer=regularizers.l2(0.002))(processed_board)
    value_batch = BatchNormalization(axis=1)(value_conv)
    value_relu = Activation('relu')(value_batch)
    value_flat = Flatten()(value_relu)
    value_hidden = Dense(
        256, activation='relu',
        kernel_regularizer=regularizers.l2(0.002))(value_flat)
    value_output = Dense(
        1, activation='tanh',
        kernel_regularizer=regularizers.l2(0.002))(value_hidden)

    model = Model(
        inputs=[board_input],
        outputs=[policy_output, value_output])

    bot = badukai.bots.zero.ZeroBot(encoder, model)
    badukai.bots.save_bot(bot, args.botfile)


if __name__ == '__main__':
    main()
