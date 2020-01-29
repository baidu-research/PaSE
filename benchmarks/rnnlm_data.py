import tensorflow.compat.v1 as tf
import tensorflow.keras as keras

class LSTMCell(keras.layers.Layer):
    def __init__(self, batch_size, num_units, **kwargs):
        self.batch_size = batch_size
        self.num_units = num_units
        self.state_size = [num_units, num_units]

        super().__init__(**kwargs)

    def build(self, input_state):
        w_shape = [2 * self.num_units, 4 * self.num_units]
        self.w = self.add_weight(shape=w_shape, initializer='uniform', name='w',
                dtype=tf.float32)
        super().build(input_state)

    def call(self, x, states):
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

    embedding = keras.layers.Embedding(params.vocab_size, params.num_units,
            input_length=params.max_seq_len)
    cells = [LSTMCell(params.batch_size, params.num_units),
            LSTMCell(params.batch_size, params.num_units)]
    rnn = keras.layers.RNN(cells, return_sequences=True, return_state=False)
    dense = keras.layers.Dense(params.vocab_size, use_bias=False)

    assert params.batch_size % num_gpus == 0
    stride = params.batch_size // num_gpus
    start, end = 0, stride
    ys = []
    for dev in devices:
        with tf.device(dev):
            ys.append(dense(rnn(embedding(inputs[start:end, ...]))))
            start = end
            end += stride
    assert start == params.batch_size

    # Loss
    with tf.device(devices[0]):
        y = tf.concat(ys, axis=0)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, y,
                reduction=tf.losses.Reduction.MEAN)
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads = opt.compute_gradients(loss, colocate_gradients_with_ops=True)
    assert all(g is not None for g in grads)
    grads = opt.apply_gradients(grads)

    return loss, grads

