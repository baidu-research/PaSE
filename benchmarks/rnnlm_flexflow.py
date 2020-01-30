import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras

import utils

class LSTMCell(keras.layers.Layer):
    def __init__(self, batch_size, num_units, device, layer, **kwargs):
        self.batch_size = batch_size
        self.num_units = num_units
        self.device = device
        self.layer = layer
        self.state_size = [num_units, num_units]
        self.counter = 0

        super().__init__(**kwargs)

    def build(self, input_state):
        w_shape = [2 * self.num_units, 4 * self.num_units]
        with tf.device(self.device):
            self.w = self.add_weight(shape=w_shape, initializer='uniform',
                    name=f'w_l{self.layer}', dtype=tf.float32)
            super().build(input_state)

    def call(self, x, states):
        with tf.device(self.exec_device):
            h, c = states
            xh = tf.concat([x, h], axis=1)

            # GEMM
            ifgo = tf.matmul(xh, self.w)

            # Apply activations
            i, f, g, o = tf.split(ifgo, 4, axis=1)
            i, f, o = [tf.sigmoid(t) for t in (i, f, o)]
            g = tf.tanh(g)

            # Elementwise ops
            c = (f * c) + (i * g)
            h = o * tf.tanh(c)
            return h, [h, c]

def model(params, inputs, labels):
    devices = params.devices
    num_gpus = len(devices)
    assert (num_gpus % 2 == 0)
    num_gpus_by_2 = num_gpus // 2

    embedding = keras.layers.Embedding(params.vocab_size, params.num_units,
            input_length=params.max_seq_len)
    dev = devices[0]
    with tf.device(dev):
        cell0 = LSTMCell(params.batch_size, params.num_units, dev, layer=0)
    dev = devices[num_gpus_by_2]
    with tf.device(dev):
        cell1 = LSTMCell(params.batch_size, params.num_units, dev, layer=1)
    cells = [cell0, cell1]
    rnn = keras.layers.RNN(cells, return_sequences=True, return_state=False)
    dense = lambda x: keras.layers.Dense(params.vocab_size // num_gpus,
            use_bias=False)(x)

    def get_next_layer_device(dev):
        idx = devices.index(dev) + num_gpus_by_2
        assert idx < num_gpus
        return devices[idx]

    with tf.device(devices[0]):
        x = embedding(inputs)

    num_data_parallel = num_gpus_by_2
    assert (params.batch_size % num_data_parallel == 0)
    stride = params.batch_size // num_data_parallel
    start, end = 0, stride
    ys = []
    for dev in devices[:num_data_parallel]:
        next_dev = get_next_layer_device(dev)
        with tf.device(dev):
            cells[0].exec_device = dev
            cells[1].exec_device = next_dev
            ys.append(rnn(x[start:end, ...]))
            start = end
            end += stride
    assert start == params.batch_size

    slices = []
    for y in ys:
        with tf.device(y.device):
            slices.append(tf.split(y, num_gpus, axis=-1))
    slices = utils.TransposeLists(slices)

    ys = []
    assert len(slices) == num_gpus
    for dev, s in zip(devices, slices):
        with tf.device(dev):
            x = tf.concat(s, axis=0)
            ys.append(dense(x))

    # Loss
    with tf.device(devices[0]):
        y = tf.concat(ys, axis=-1)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, y,
                reduction=tf.losses.Reduction.MEAN)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads = opt.compute_gradients(loss, colocate_gradients_with_ops=True)
    assert all(g is not None for g in grads)
    grads = opt.apply_gradients(grads)

    return loss, grads

