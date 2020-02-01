import mesh_tensorflow.placement_mesh_impl as mtf
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import string
import utils

def parallel_matmul(x, hs, ws, k_dim):
    wlen = len(ws)
    wlen_by_2 = wlen // 2
    assert wlen % 2 == 0
    assert len(hs) == wlen_by_2

    assert (k_dim % wlen_by_2 == 0)
    stride = k_dim // wlen_by_2

    start, end = 0, stride
    xs = []
    for w in ws[:wlen_by_2]:
        with tf.device(w.device):
            xs.append(tf.matmul(x[:, start:end], w))
            start = end
            end += stride
    assert start == k_dim

    _hs = []
    for h, w in zip(hs, ws[wlen_by_2:]):
        assert ((not h.device) or (h.device == w.device))
        with tf.device(w.device):
            _hs.append(tf.matmul(h, w))

    return mtf.allreduce_ring(xs + _hs, [w.device for w in ws])

class LSTMCell(keras.layers.Layer):
    def __init__(self, batch_size, num_units, devices, k, n, layer, **kwargs):
        assert (k % 2 == 0)
        self.num_gpus = len(devices)
        self.k = k
        self.n = n
        self.batch_size = batch_size
        self.num_units = num_units
        self.layer = layer
        self.num_states = (k // 2) * n

        h_state_size = [num_units // (k // 2)] * self.num_states
        c_state_size = [num_units // n] * self.num_gpus
        self.state_size = h_state_size + c_state_size

        self.devices = devices
        super().__init__(**kwargs)

    def build(self, input_state):
        self.ws = [[] for _ in range(self.n)]
        for i, d in enumerate(self.devices):
            with tf.device(d):
                w_shape = [(2 * self.num_units) // self.k, 
                        (4 * self.num_units) // self.n]
                w = self.add_weight(shape=w_shape, initializer='uniform',
                        dtype=tf.float32, name=f'w_l{self.layer}_{i}')
                self.ws[i % self.n].append(w)
        assert all(len(w) == len(self.ws[0]) for w in self.ws[1:])
        super().build(input_state)

    def call(self, x, states):
        assert len(states) == (self.num_states + self.num_gpus)
        hs, cs = states[:self.num_states], states[self.num_states:]

        num_hs = self.k // 2
        hs = [hs[i:i+num_hs] for i in range(0, self.num_states, num_hs)]
        cs = [cs[i:i+self.k] for i in range(0, self.num_gpus, self.k)]
        assert len(hs) == self.n
        assert len(cs) == self.n

        # GEMM
        ifgos = []
        for h, w in zip(hs, self.ws):
            assert (2 * len(h)) == len(w)
            ifgos.append(parallel_matmul(x, h, w, self.num_units))
        assert len(ifgos) == self.n
        assert all(len(ifgo) == self.k for ifgo in ifgos)

        new_cs = []
        new_hs_transpose = []
        for _ifgo, _c in zip(ifgos, cs):
            assert len(_ifgo) == len(_c)
            _new_cs = []
            _new_hs = []
            for ifgo, c in zip(_ifgo, _c):
                assert ((not c.device) or (ifgo.device == c.device))
                with tf.device(ifgo.device):
                    # Apply activations
                    i, f, g, o = tf.split(ifgo, 4, axis=1)
                    i, f, o = [tf.sigmoid(t) for t in (i, f, o)]
                    g = tf.tanh(g)

                    # Elementwise ops
                    c = (f * c) + (i * g)
                    h = o * tf.tanh(c)
                    _new_cs.append(c)
                    _new_hs.append(h)
            new_cs += _new_cs
            new_hs_transpose.append(_new_hs)
        assert len(new_cs) == self.num_gpus
        assert len(new_hs_transpose) == self.n
        assert all(len(h) == self.k for h in new_hs_transpose)

        # Concatenate 'n' splits of 'h', and re-distribute to 'k' devices
        new_hs_transpose = utils.TransposeLists(new_hs_transpose)
        new_hs_full = [mtf.allconcat_ring(hs, [h.device for h in hs], 1)
                for hs in new_hs_transpose]
        new_hs_full = utils.TransposeLists(new_hs_full)
        assert len(new_hs_full) == self.n
        assert all(len(h) == self.k for h in new_hs_full)

        x = new_hs_full[0][0]
        assert x.device == self.devices[0]

        new_hs = []
        for hs in new_hs_full:
            stride = self.num_units // num_hs
            start, end  = 0, stride
            _new_hs = []
            for h in hs[num_hs:]:
                with tf.device(h.device):
                    _new_hs.append(h[:, start:end])
                    start = end
                    end += stride
            assert start == self.num_units
            new_hs += _new_hs

        return x, [new_hs, new_cs]

def model(params, inputs, labels):
    devices = params.devices
    num_gpus = len(devices)

    if num_gpus == 8:
        n, k = 4, 2
    elif num_gpus == 16:
        n, k = 4, 4
    elif num_gpus == 32:
        n, k = 8, 4
    else:
        assert False

    devices = [devices[i:i+n] for i in range(0, num_gpus, n)]
    devices = utils.FlattenList(utils.TransposeLists(devices))
    assert set(devices) == set(params.devices)

    cells = [LSTMCell(params.batch_size, params.num_units, devices, k, n,
                layer=0),
            LSTMCell(params.batch_size, params.num_units, devices, k, n,
                layer=1)]
    rnn = keras.layers.RNN(cells, return_sequences=True, return_state=False)
    dense = lambda x: keras.layers.Dense(params.vocab_size // num_gpus,
            use_bias=False)(x)

    # Model
    it = iter(devices)
    device_selector = lambda _: next(it)
    with tf.device(device_selector):
        embedding_weights = tf.get_variable('embed_weights',
                shape=[params.vocab_size, params.num_units],
            partitioner=tf.fixed_size_partitioner(num_gpus, axis=0))
    xs = rnn(tf.nn.embedding_lookup(embedding_weights, inputs))

    # Final dense layer
    ys = []
    for dev in devices:
        with tf.device(dev):
            ys.append(dense(xs))
    with tf.device(devices[0]):
        y = tf.concat(ys, axis=2)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, y,
                reduction=tf.losses.Reduction.MEAN)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads = opt.compute_gradients(loss, colocate_gradients_with_ops=True)
    assert all(g is not None for g in grads)
    grads = opt.apply_gradients(grads)

    return loss, grads

