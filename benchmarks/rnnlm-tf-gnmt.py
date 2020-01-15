import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras

import math
import datetime
import sys, time, os
import string, random
import argparse
import functools

import common
from dataloader import TextDataLoader

class Params():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_units = 2048
        self.max_seq_len = 256
        self.num_layers = 2


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

def main():
    parser = argparse.ArgumentParser(formatter_class =
            argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocab', type=str, help="Source vocab data file.")
    parser.add_argument('--text', type=str, help="Source text data file.")
    parser.add_argument('--max_steps', type=int, required=False, default=50,
            help='Maximum no. of steps to execute')

    trainer = common.Trainer(parser)
    args = trainer.args
    params = Params(args.batch_size)
    servername = 'localhost' if trainer.num_nodes == 1 else 'worker'
    devices = [f'/job:{servername}/replica:0/task:{i}/device:GPU:{j}' for i in
            range(trainer.num_nodes) for j in range(args.gpus)]
    
    # Initialize dataset
    dataset = TextDataLoader(args.batch_size, args.vocab, None, args.text, None,
            max_seq_len=params.max_seq_len)
    inputs, labels, _, _ = dataset.next_batch()

    with open(args.vocab) as f:
        for vocab_size, _ in enumerate(f):
            pass
    vocab_size = int(math.ceil(vocab_size / 8)) * int(8)
    print("Vocab size: %d" % vocab_size)

    assert trainer.num_gpus % 2 == 0
    num_gpus_by_2 = trainer.num_gpus // 2

    # Model
    dev = devices[0]
    with tf.device(dev):
        cell0 = LSTMCell(params.batch_size, params.num_units, dev, layer=0)
    dev = devices[num_gpus_by_2]
    with tf.device(dev):
        cell1 = LSTMCell(params.batch_size, params.num_units, dev, layer=1)
    cells = [cell0, cell1]
    rnn = keras.layers.RNN(cells, return_sequences=True, return_state=False)
    dense = keras.layers.Dense(vocab_size, use_bias=False)

    def get_next_layer_device(dev):
        dev = dev.split(':')
        dev_id = int(dev[-1]) + num_gpus_by_2
        assert dev_id < trainer.num_gpus

        dev[-1] = str(dev_id)
        dev = ':'.join(dev)
        return dev

    num_data_parallel = num_gpus_by_2
    xs = tf.split(inputs, num_data_parallel, axis=0)
    embedding_weights = tf.get_variable('embed_weights', shape=[vocab_size,
        params.num_units])
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

    # Train
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    config = tf.ConfigProto(log_device_placement=False,
            allow_soft_placement=True)
    trainer.train(tf.global_variables_initializer(), loss, [grads], dataset,
            train_batches_per_epoch=args.max_steps, config=config,
            run_options=run_options)


if __name__ == '__main__':
    main()

