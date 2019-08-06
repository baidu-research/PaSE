import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import math
import datetime
import sys, time, os
import string, random
import argparse
import functools

from dataloader import TextDataLoader

class Params():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_units = 2048
        self.max_seq_len = 512
        self.num_layers = 2


class LSTMCell(keras.layers.Layer):
    def __init__(self, batch_size, num_units, devices, num_k_splits,
            num_n_splits, **kwargs):
        self.num_k_splits = num_k_splits
        self.num_n_splits = num_n_splits
        self.batch_size = batch_size
        self.num_units = num_units

        h_state_size = [num_units // self.num_k_splits] * self.num_k_splits
        c_state_size = [num_units // self.num_n_splits] * self.num_n_splits
        self.state_size = h_state_size + c_state_size

        self.devices = devices
        super().__init__(**kwargs)

    def build(self, input_state):
        self.ws = []
        for d in self.devices:
            with tf.device(d):
                w_shape = [(2 * self.num_units) // self.num_k_splits, (4 *
                    self.num_units) // self.num_n_splits]
                w = self.add_weight(shape=w_shape, initializer='uniform',
                        dtype=tf.float32, name=f'w_{d.device_index}')
                self.ws.append(w)
        super().build(input_state)

    def call(self, x, states):
        assert len(states) == self.num_k_splits + self.num_n_splits
        hs, cs = states[:self.num_k_splits], states[self.num_k_splits:]

        assert x.shape.as_list() == [self.batch_size, self.num_units]
        xs = tf.split(x, self.num_k_splits, axis=1)

        # Concatenate x and h
        xhs = []
        for x, h in zip(xs, hs):
            with tf.device(h.device):
                xhs.append(tf.concat([x, h], axis=1))

        # Copy xh to all devices
        xhs_copies = []
        for i, xh in enumerate(xhs):
            for j in range(self.num_n_splits):
                dev_idx = i*self.num_n_splits+j
                with tf.device(self.devices[dev_idx]):
                    xhs_copies.append(tf.identity(xh))
        assert len(xhs_copies) == 8
        assert all(xh.device == device.to_string() for xh, device in
                zip(xhs_copies, self.devices))

        # GEMM
        partial_ifgos = []
        for xh, w in zip(xhs_copies, self.ws):
            assert xh.device == w.device
            with tf.device(w.device):
                partial_ifgos.append(tf.matmul(xh, w))

        # Reduce-sum partial GEMMs
        if self.num_k_splits > 1:
            ifgo_list = [[] for _ in range(self.num_n_splits)]
            for i, ifgo in enumerate(partial_ifgos):
                ifgo_list[i % self.num_n_splits].append(ifgo)
            ifgos = []
            for ifgo in ifgo_list:
                with tf.device(ifgo[0].device):
                    ifgos.append(tf.add_n(ifgo))
        else:
            ifgos = partial_ifgos
        assert len(ifgos) == self.num_n_splits
        assert all(ifgo.device == device.to_string() for ifgo, device in
                zip(ifgos, self.devices))

        # Apply activations
        new_hs_split, new_cs = [], []
        for ifgo, c in zip(ifgos, cs):
            with tf.device(ifgo.device):
                i, f, g, o = tf.split(ifgo, 4, axis=1)
                i, f, o = [tf.sigmoid(t) for t in (i, f, o)]
                g = tf.tanh(g)

                c = (f * c) + (i * g)
                h = o * tf.tanh(c)
                new_cs.append(c)
                new_hs_split.append(h)

        # Concatenate hs
        if self.num_n_splits > self.num_k_splits:
            num_concats = self.num_n_splits // self.num_k_splits
            new_hs_split = [new_hs_split[i:i+num_concats] for i in range(0,
                len(new_hs_split), num_concats)]
            new_hs = []
            for d, hs in zip(self.devices[::self.num_n_splits], new_hs_split):
                with tf.device(d):
                    new_hs.append(tf.concat(hs, axis=1))
            assert [h.device for h in new_hs] == [d.to_string() for d in
                    self.devices[::self.num_n_splits]]
        elif self.num_n_splits < self.num_k_splits:
            new_hs = []
            num_splits = self.num_k_splits // self.num_n_splits
            for h in new_hs_split:
                with tf.device(h.device):
                    new_hs += tf.split(h, num_splits, axis=1)
        else:
            new_hs = new_hs_split

        assert [c.device for c in new_cs] == [d.to_string() for d in
                self.devices[:self.num_n_splits]]

        with tf.device(self.devices[0]):
            x = tf.concat(new_hs, axis=1)
        return x, new_hs + new_cs


def main():
    parser = argparse.ArgumentParser(formatter_class =
            argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b', '--batch', type=int, required=False, default=64,
            help="Batch size")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=3,
            help="No. of epochs")
    parser.add_argument('--max_steps', type=int, required=False, default=50,
            help='Maximum no. of steps to execute')
    parser.add_argument('--display_steps', type=int, required=False, default=10,
            help="No. of epochs")
    parser.add_argument('--vocab', type=str, help="Source vocab data file.")
    parser.add_argument('--text', type=str, help="Source text data file.")
    args = parser.parse_args()
    params = Params(args.batch)
    [print(f'{arg} : {val}') for arg, val in vars(args).items()]

    if args.procs != 8:
        raise NotImplementedError('Current implementation only handles 8 GPUs.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(args.procs))[:-1]
    devices = [tf.DeviceSpec(device_type='GPU', device_index=i) for i in
            range(args.procs)]
    
    # Initialize dataset
    dataset = TextDataLoader(args.batch, args.vocab, None, args.text, None,
            max_seq_len=params.max_seq_len)
    inputs, labels, _, _ = dataset.next_batch()

    with open(args.vocab) as f:
        for vocab_size, _ in enumerate(f):
            pass
    vocab_size = int(math.ceil(vocab_size / 8)) * int(8)
    print("Vocab size: %d" % vocab_size)

    # RNN
    num_k_splits = 2
    num_n_splits = 4
    cells = [LSTMCell(params.batch_size, params.num_units, devices,
        num_k_splits, num_n_splits),
            LSTMCell(params.batch_size, params.num_units, devices, num_k_splits,
                num_n_splits)]
    rnn = keras.layers.RNN(cells, return_sequences=True, return_state=False)

    # Initial states
    hs0, hs1 = [], []
    for device in devices[::num_n_splits]:
        with tf.device(device):
            hs0.append(tf.zeros([params.batch_size, params.num_units //
                num_k_splits], dtype=tf.float32))
            hs1.append(tf.zeros([params.batch_size, params.num_units //
                num_k_splits], dtype=tf.float32))
    cs0, cs1 = [], []
    for device in devices[:num_n_splits]:
        with tf.device(device):
            cs0.append(tf.zeros([params.batch_size, params.num_units //
                num_n_splits], dtype=tf.float32))
            cs1.append(tf.zeros([params.batch_size, params.num_units //
                num_n_splits], dtype=tf.float32))
    states = hs0 + cs0 + hs1 + cs1

    def get_device_fn():
        i = 0
        def fn(_):
            nonlocal i
            j = i
            i = (i + 1) % len(devices)
            return devices[j]
        return fn

    # Model
    with tf.device(get_device_fn()):
        embedding_weights = tf.get_variable('embed_weights', shape=[vocab_size,
            params.num_units], partitioner=tf.fixed_size_partitioner(8, axis=0))
    embed = tf.nn.embedding_lookup(embedding_weights, inputs)
    xs = rnn(embed, initial_state=states)

    # Final dense layer
    ys = []
    for dev in devices:
        with tf.device(dev):
            ys.append(keras.layers.Dense(vocab_size // 8, use_bias=False)(xs))
    with tf.device(devices[0]):
        y = tf.concat(ys, axis=2)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, y,
                reduction=tf.losses.Reduction.MEAN)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads = opt.compute_gradients(loss, colocate_gradients_with_ops=True)
    assert all(g is not None for g in grads)
    grads = opt.apply_gradients(grads)

    cnt = 0
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    config = tf.ConfigProto(#log_device_placement=True,
            allow_soft_placement=False)
    with tf.Session(config=config) as sess:
        dataset.reset_pointer()
        sess.run(tf.global_variables_initializer())

        tot_time = float(0)
        start = time.time()
        for epoch in range(args.epochs):
            step = 0

            while True:
                try:
                    loss_val, *_ = sess.run([loss, grads], options=run_options)
                    cnt += 1
                    step += 1
                    if step > args.max_steps:
                        break
                except tf.errors.OutOfRangeError:
                    break

                if step % args.display_steps == 0 and step > 0:
                    print("Epoch: " + str(epoch) + "; Loss: " + str(loss_val))

            dataset.reset_pointer()
        end = time.time()
        tot_time += (end - start)

    samples_per_sec = (args.batch * params.max_seq_len * cnt) / tot_time
    print("Throughput: " + str(samples_per_sec) + " samples / sec")



if __name__ == '__main__':
    main()

