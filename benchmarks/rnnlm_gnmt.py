import tensorflow.compat.v1 as tf
import tensorflow.keras as keras

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
                    name='w', dtype=tf.float32)
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

    dev = devices[0]
    with tf.device(dev):
        cell0 = LSTMCell(params.batch_size, params.num_units, dev, layer=0)
    dev = devices[num_gpus_by_2]
    with tf.device(dev):
        cell1 = LSTMCell(params.batch_size, params.num_units, dev, layer=1)
    cells = [cell0, cell1]
    rnn = keras.layers.RNN(cells, return_sequences=True, return_state=False)
    dense = keras.layers.Dense(params.vocab_size, use_bias=False)

    def get_next_layer_device(dev):
        dev = dev.split(':')
        dev_id = int(dev[-1]) + num_gpus_by_2
        assert dev_id < num_gpus

        dev[-1] = str(dev_id)
        dev = ':'.join(dev)
        return dev

    num_data_parallel = num_gpus_by_2
    xs = tf.split(inputs, num_data_parallel, axis=0)
    embedding_weights = tf.get_variable('embed_weights',
            shape=[params.vocab_size, params.num_units])
    ys = []
    for x, dev in zip(xs, devices):
        with tf.device(dev):
            x = tf.nn.embedding_lookup(embedding_weights, x)
            cells[0].exec_device = dev
            cells[1].exec_device = get_next_layer_device(dev)
            ys.append(dense(rnn(x)))
    y = tf.concat(ys, axis=0)

    # Loss
    with tf.device(devices[num_gpus_by_2]):
        loss = tf.losses.sparse_softmax_cross_entropy(labels, y,
                reduction=tf.losses.Reduction.MEAN)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads = opt.compute_gradients(loss, colocate_gradients_with_ops=True)
    assert all(g is not None for g in grads)
    grads = opt.apply_gradients(grads)

    return loss, grads

