from keras.layers import Activation, BatchNormalization, Conv2D


def conv_unit(filter_size, num_filters, input_shape=None):
    return [
        Conv2D(
            num_filters, (filter_size, filter_size),
            padding='same',
            data_format='channels_first'),
        BatchNormalization(axis=1),
        Activation('relu'),
    ]


def layers(input_shape):
    return \
        conv_unit(3, 64, input_shape=input_shape) + \
        conv_unit(3, 64) + \
        conv_unit(3, 64)
