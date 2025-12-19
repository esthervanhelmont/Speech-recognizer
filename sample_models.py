from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)


def simple_rnn_model(input_dim, output_dim=29):
    """Build a simple recurrent network for speech."""
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True,
                   implementation=2, name='rnn')(input_data)
    # Softmax over characters
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def rnn_model(input_dim, units, activation, output_dim=29):
    """Build a recurrent network with BatchNorm + TimeDistributed Dense."""
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Recurrent layer
    simp_rnn = GRU(units,
                   activation=activation,
                   return_sequences=True,
                   implementation=2,
                   name='rnn')(input_data)
    # Batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TimeDistributed Dense over each time step
    time_dense = TimeDistributed(Dense(output_dim),
                                 name='time_dense')(bn_rnn)
    # Softmax
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
                  conv_border_mode, units, output_dim=29):
    """Build a convolutional + recurrent network for speech."""
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # 1D convolution
    conv_1d = Conv1D(filters,
                     kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # BatchNorm on conv output
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Recurrent layer (use GRU instead of SimpleRNN to avoid exploding gradients)
    simp_rnn = GRU(units,
                   return_sequences=True,
                   implementation=2,
                   name='rnn')(bn_cnn)
    # BatchNorm on RNN output
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TimeDistributed Dense
    time_dense = TimeDistributed(Dense(output_dim),
                                 name='time_dense')(bn_rnn)
    # Softmax
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_output_length(input_length, filter_size, border_mode, stride,
                      dilation=1):
    """Compute the length of the output sequence after 1D convolution."""
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """Build a deep (stacked) recurrent network for speech."""
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Stacked GRU layers, each followed by BatchNorm
    x = input_data
    for i in range(recur_layers):
        x = GRU(units,
                return_sequences=True,
                implementation=2,
                name=f'rnn_{i + 1}')(x)
        x = BatchNormalization(name=f'bn_rnn_{i + 1}')(x)

    # TimeDistributed Dense
    time_dense = TimeDistributed(Dense(output_dim),
                                 name='time_dense')(x)
    # Softmax
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """Build a bidirectional recurrent network for speech."""
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Bidirectional GRU
    bidir_rnn = Bidirectional(
        GRU(units,
            return_sequences=True,
            implementation=2,
            name='rnn'),
        name='bidir_rnn')(input_data)
    # TimeDistributed Dense
    time_dense = TimeDistributed(Dense(output_dim),
                                 name='time_dense')(bidir_rnn)
    # Softmax
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def final_model(input_dim=161, output_dim=29):
    """Build a deeper final model: CNN + Bi-GRU + BatchNorm + TimeDistributed."""
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Hyperparameters (simple, not over-engineered)
    filters = 200
    kernel_size = 11
    conv_stride = 2
    conv_border_mode = 'valid'
    rnn_units = 200

    # Convolution front-end
    conv_1d = Conv1D(filters,
                     kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    # Bidirectional GRU
    bidir_rnn = Bidirectional(
        GRU(rnn_units,
            return_sequences=True,
            implementation=2,
            name='rnn'),
        name='bidir_rnn')(bn_cnn)
    bn_rnn = BatchNormalization(name='bn_rnn')(bidir_rnn)

    # TimeDistributed Dense over time steps
    time_dense = TimeDistributed(Dense(output_dim),
                                 name='time_dense')(bn_rnn)
    # Softmax
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Model
    model = Model(inputs=input_data, outputs=y_pred)
    # Output length after the convolution
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)

    print(model.summary())
    return model
