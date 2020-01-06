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

num_data_parallel = 2
class Params():
    def __init__(self):
        self.num_units = 1024
        self.max_seq_len = 256
        self.num_layers = 4


class LSTMCell(keras.layers.Layer):
    def __init__(self, num_units, layer, num_layers,
            device, attention=False, **kwargs):
        self.num_units = num_units
        self.layer = layer
        self.num_layers = num_layers
        self.attention = attention
        self.device = device

        self.state_size = [num_units, num_units]
        if attention and layer == 0:
            self.state_size.append(num_units)
        self.output_size = num_units
        if attention and layer != num_layers - 1:
            self.output_size += num_units

        self.counter = 0
        super().__init__(**kwargs)

    def build(self, input_shape):
        with tf.device(self.device):
            w_shape = [(2 + self.attention) * self.num_units, 4 *
                    self.num_units]
            self.w = self.add_weight(shape=w_shape, initializer='uniform',
                    name='w', dtype=tf.float32)
            super().build(input_shape)

    def call(self, x, states, constants):
        device_name = x.device.split(':')
        device_name[-1] = str(int(self.counter % num_data_parallel) 
                + (self.layer * num_data_parallel))
        device_name = ':'.join(device_name)
        self.counter += 1
        with tf.device(device_name):
            h, c = states[:2]
            concat_lst = [h, x]
            if self.attention:
                # For layer 0, context is computed and stored in states
                # For other layers, context is passed from previous layer, by
                # concatenating with 'x'
                if self.layer == 0:
                    assert len(constants) == 2
                    assert len(states) == 3
                    assert x.shape.as_list()[-1] == self.num_units
                    curr_context = states[2]
                    concat_lst.append(curr_context)
                else:
                    assert len(states) == 2
                    assert x.shape.as_list()[-1] == 2 * self.num_units
                    curr_context = tf.split(x, 2, axis=1)[1]

            def Attention(h):
                enc_attn, enc_out = constants
                assert len(enc_attn.shape.as_list()) == 3
                assert enc_attn.shape[0] == h.shape[0]

                score = tf.einsum('ble,be->bl', enc_attn, h)
                score = tf.nn.softmax(score, axis=-1)
                context = tf.einsum('bl,ble->be', score, enc_out)
                return context

            xh = tf.concat(concat_lst, axis=1)

            # GEMM
            ifgo = tf.matmul(xh, self.w)

            # Apply activations
            i, f, g, o = tf.split(ifgo, 4, axis=1)
            i, f, o = [tf.sigmoid(t) for t in (i, f, o)]
            g = tf.tanh(g)

            # Elementwise ops
            c = (f * c) + (i * g)
            h = o * tf.tanh(c)

            y = h
            states = [h, c]

            # For layer 0, compute context for next timestep and add it to the
            # states. Except for the last layer, all other layers concatenate
            # 'context' with 'x' and  pass it to next layer.
            if self.attention:
                if self.layer == 0:
                    next_context = Attention(h)
                    states.append(next_context)

                if self.layer != self.num_layers - 1:
                    y = tf.concat([y, curr_context], axis=1)

            return y, states

class RNN(keras.layers.RNN):
    def build(self, input_shapes):
        assert self.constants_spec is not None
        constants_shape = [spec.shape for spec in self.constants_spec]
        input_shapes = [input_shapes] + constants_shape
        super().build(input_shapes)

class Model():
    def __init__(self, args, params, src_vocab_size, tgt_vocab_size, devices,
            num_data_parallel):
        self.num_units = params.num_units
        self.num_layers = params.num_layers

        # Encoder
        self.enc_weights = tf.get_variable('encoder_weights',
                shape=[src_vocab_size, params.num_units])

        # Encoder RNN + attention
        cells = []
        for i, d in enumerate(devices[::num_data_parallel]):
            with tf.device(d):
                cells.append(LSTMCell(params.num_units, i, params.num_layers,
                    d))
        self.enc_rnn = RNN(cells, return_sequences=True, return_state=True)
        self.enc_attn = keras.layers.Dense(params.num_units, use_bias=False)

        # Decoder
        self.dec_weights = tf.get_variable('decoder_weights',
                shape=[tgt_vocab_size, params.num_units])

        # Decoder RNN
        cells = []
        for i, d in enumerate(devices[::num_data_parallel]):
            with tf.device(d):
                cells.append(LSTMCell(params.num_units, i, params.num_layers, d,
                    True))
        self.dec_rnn = RNN(cells, return_sequences=True, return_state=False)

        # Final projection
        self.proj = keras.layers.Dense(tgt_vocab_size, use_bias=False)

    def __call__(self, device, enc_inp, dec_inp):
        with tf.device(device):
            # Encoder
            x = tf.nn.embedding_lookup(self.enc_weights, enc_inp)
            enc_out, *states = self.enc_rnn(x, constants=tf.zeros(1))
            enc_attn = self.enc_attn(enc_out)
            constants = [enc_attn, enc_out]

            # Decoder
            x = tf.nn.embedding_lookup(self.dec_weights, dec_inp)
            context = tf.zeros([dec_inp.shape.as_list()[0], self.num_units],
                    dtype=tf.float32)
            states.insert(2, context)
            dec_out = self.dec_rnn(x, initial_state=states, constants=constants)
            dec_out = self.proj(dec_out)
            
            return dec_out

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
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(3)), 
            help="Strategy to use. 0: DataParallel, \
                    1: Optimized, \
                    2: Expert designed")
    parser.add_argument('--src_vocab', type=str, help="Source vocab data file.")
    parser.add_argument('--tgt_vocab', type=str, help="Target vocab data file.")
    parser.add_argument('--src_text', type=str, help="Source text data file.")
    parser.add_argument('--tgt_text', type=str, help="Target text data file.")
    args = parser.parse_args()
    params = Params()
    [print(f'{arg} : {val}') for arg, val in vars(args).items()]

    if args.procs != 8:
        raise NotImplementedError('Current implementation only handles 8 GPUs.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(args.procs))[:-1]
    devices = [tf.DeviceSpec(device_type='GPU', device_index=i) for i in
            range(args.procs)]
    
    # Initialize dataset
    dataset = TextDataLoader(args.batch, args.src_vocab, args.tgt_vocab,
            args.src_text, args.tgt_text, max_seq_len = params.max_seq_len)
    src_pad_id = tf.cast(dataset.src_pad_id, tf.int32)
    enc_inputs, dec_inputs, _, _ = dataset.next_batch()

    with open(args.src_vocab) as f:
        for src_vocab_size, _ in enumerate(f):
            pass
    with open(args.tgt_vocab) as f:
        for tgt_vocab_size, _ in enumerate(f):
            pass
    src_vocab_size = int(math.ceil(src_vocab_size / 8)) * int(8)
    tgt_vocab_size = int(math.ceil(tgt_vocab_size / 8)) * int(8)
    print("Source vocab size: %d" % src_vocab_size)
    print("Target vocab size: %d" % tgt_vocab_size)

    model = Model(args, params, src_vocab_size, tgt_vocab_size, devices,
            num_data_parallel)
    enc_inps = tf.split(enc_inputs, num_data_parallel, axis=0)
    dec_inps = tf.split(dec_inputs, num_data_parallel, axis=0)
    ys = []
    for model_args in zip(devices, enc_inps, dec_inps):
        ys.append(model(*model_args))

    # Loss
    with tf.device(devices[0]):
        y = tf.concat(ys, axis=0)
        loss = tf.losses.sparse_softmax_cross_entropy(dec_inputs, y,
                reduction=tf.losses.Reduction.MEAN)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads = opt.compute_gradients(loss, colocate_gradients_with_ops=True)
    assert all(g is not None for g in grads)
    grads = opt.apply_gradients(grads)
    init_ops = tf.global_variables_initializer()

    cnt = 0
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    config = tf.ConfigProto(log_device_placement=False,
            allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        dataset.reset_pointer()
        sess.run(init_ops)

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

