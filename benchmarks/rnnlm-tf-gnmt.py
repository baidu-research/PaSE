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
    def __init__(self, batch_size, num_units, device, layer, **kwargs):
        self.batch_size = batch_size
        self.num_units = num_units
        self.device = device
        self.layer = layer
        self.state_size = [num_units, num_units]

        super().__init__(**kwargs)

    def build(self, input_state):
        w_shape = [2 * self.num_units, 4 * self.num_units]
        with tf.device(self.device):
            self.w = self.add_weight(shape=w_shape, initializer='uniform',
                    name='w', dtype=tf.float32)
            super().build(input_state)

    def call(self, x, states):
        device_name = x.device.split(':')
        device_name[-1] = str(int(device_name[-1]) + (self.layer * 4))
        device_name = ':'.join(device_name)
        with tf.device(device_name):
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

    # Model
    model = keras.Sequential()
    with tf.device(devices[0]):
        cell0 = LSTMCell(params.batch_size, params.num_units, devices[0],
                layer=0)
    with tf.device(devices[4]):
        cell1 = LSTMCell(params.batch_size, params.num_units, devices[4],
                layer=1)
    cells = [cell0, cell1]
    model.add(keras.layers.RNN(cells, return_sequences=True,
        return_state=False))
    model.add(keras.layers.Dense(vocab_size, use_bias=False))

    num_data_parallel = 4
    xs = tf.split(inputs, num_data_parallel, axis=0)
    embedding_weights = tf.get_variable('embed_weights', shape=[vocab_size,
        params.num_units])
    ys = []
    for x, dev in zip(xs, devices):
        with tf.device(dev):
            x = tf.nn.embedding_lookup(embedding_weights, x)
            ys.append(model(x))
    y = tf.concat(ys, axis=0)

    # Loss
    with tf.device(devices[4]):
        loss = tf.losses.sparse_softmax_cross_entropy(labels, y,
                reduction=tf.losses.Reduction.MEAN)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads = opt.compute_gradients(loss, colocate_gradients_with_ops=True)
    assert all(g is not None for g in grads)
    grads = opt.apply_gradients(grads)

    cnt = 0
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    config = tf.ConfigProto(log_device_placement=False,
            allow_soft_placement=True)
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

    samples_per_sec = (args.batch * cnt) / tot_time
    print("Throughput: " + str(samples_per_sec) + " samples / sec")



if __name__ == '__main__':
    main()

