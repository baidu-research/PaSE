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

def Devices(lst):
    return [x.device for x in lst]

def FlattenList(l):
   return [item for sublist in l for item in sublist]

class Params():
    def __init__(self, batch_size, max_seq_len):
        self.batch_size = batch_size
        self.num_units = 2048
        self.max_seq_len = max_seq_len
        self.num_layers = 2

def AllConcatRing(xs, axis, devices=None):
  devices = devices or [x.device for x in xs]
  n = len(xs)
  assert len(devices) == n
  if n == 1:
    return xs

  # [target, source]
  parts = [[xs[target] if target == source else None for source in range(n)]
           for target in range(n)]
  for distance in range(1, n // 2 + 1):
    for target in range(n):
      source = (target + distance) % n
      if parts[target][source] is None:
        with tf.device(devices[target]):
          parts[target][source] = tf.identity(parts[(target + 1) % n][source])
      source = (target - distance) % n
      if parts[target][source] is None:
        with tf.device(devices[target]):
          parts[target][source] = tf.identity(parts[(target - 1) % n][source])

  ys = []
  for i, dev in enumerate(devices):
      with tf.device(dev):
          ys.append(tf.concat(parts[i], axis=axis))
  return ys

def split_inputs(xs, devices):
    device_ids = [d.device_index for d in devices]
    d_spec0, d_spec1, d0, d1, xs0, xs1 = [], [], [], [], [], []
    for x, spec, d in zip(xs, devices, device_ids):
        if d < 4:
            d0.append(d)
            xs0.append(x)
            d_spec0.append(spec)
        else:
            d1.append(d)
            xs1.append(x)
            d_spec1.append(spec)

    flag = (not d0) or (not d1) \
            or (len(d0) == len(d1) 
                    and all(i+4 == j and device_ids.index(i) <
                        device_ids.index(j) for i, j in zip(d0, d1)))
    return d_spec0, d_spec1, d0, d1, xs0, xs1, flag


def AllConcat(xs, axis, devices=None):
    devices = devices or Devices(xs)
    if len(xs) == 1:
        return xs
    ys = []
    for x, d in zip(xs, devices):
        with tf.device(d):
            ys.append(tf.concat([x]*len(xs), axis))
    return ys

    d_spec0, d_spec1, d0, d1, xs0, xs1, flag = split_inputs(xs, devices)
    if not flag: # Fallback to ring version
        return AllConcatRing(xs, axis, devices)

    def concat_fn(xs, d_spec):
        if len(xs) <= 1:
            return xs
        ys = []
        for d in d_spec:
            with tf.device(d):
                ys.append(tf.concat(xs, axis))
        return ys

    ys0 = concat_fn(xs0, d_spec0)
    ys1 = concat_fn(xs1, d_spec1)

    if xs0 and xs1:
        tmp_ys = [y for y in zip(ys0*2, ys1*2)]
        ys = []
        for d in devices:
            with tf.device(d):
                ys.append(tf.concat(tmp_ys, axis))
    else:
        ys = ys0 + ys1
    assert len(xs) == len(ys)
    assert all(y.shape[axis] == x.shape[axis] * len(devices) for x, y in zip(xs,
        ys))
    return ys

def ParallelGEMM(As, Bs, eq=None):
    assert len(As) == len(Bs)
    partial_sums = []
    if eq is None:
        eq = string.ascii_letters[:As[0].shape.ndims] + ',' \
                + string.ascii_letters[As[0].shape.ndims-1
                        :As[0].shape.ndims+Bs[0].shape.ndims-1]
    for A, B in zip(As, Bs):
        assert A.device == B.device
        with tf.device(A.device):
            partial_sums.append(tf.einsum(eq, A, B))
    with tf.device(As[0].device):
        y = tf.add_n(partial_sums)
    return y

def ParallelDense(x, w_shape, num_k_splits, num_n_splits, devices, name,
        concat=False):
    assert len(w_shape) == 2
    assert num_k_splits * num_n_splits <= len(devices)

    if isinstance(x, list):
        assert len(x) == num_k_splits
        xs = x
    else:
        xs = tf.split(x, num_k_splits, axis=-1)
    ys = []
    for i in range(num_n_splits):
        sums = []
        for j, x in enumerate(xs):
            ndims = x.shape.ndims
            idx = j*num_n_splits+i
            dev = devices[idx]
            eq = string.ascii_letters[:ndims] + ',' \
                    + string.ascii_letters[ndims-1:ndims+1]
            with tf.device(dev):
                w = tf.get_variable(f'{name}_{idx}', shape=[w_shape[0] //
                    num_k_splits, w_shape[1] // num_n_splits])
                sums.append(tf.einsum(eq, x, w))
        with tf.device(devices[i]):
            ys.append(tf.add_n(sums))
    return ys if not concat else tf.concat(ys, axis=-1)


class LSTMCell(keras.layers.Layer):
    def __init__(self, batch_size, num_units, devices, num_k_splits,
            num_n_splits, **kwargs):
        self.num_gpus = len(devices)
        self.num_k_splits = num_k_splits
        self.num_n_splits = num_n_splits
        self.batch_size = batch_size
        self.num_units = num_units

        h_state_size = [num_units // self.num_k_splits] * self.num_gpus
        c_state_size = [num_units // self.num_n_splits] * self.num_n_splits
        self.state_size = h_state_size + c_state_size

        self.devices = devices
        super().__init__(**kwargs)

    def build(self, input_state):
        self.ws = []
        for i, d in enumerate(self.devices):
            with tf.device(d):
                w_shape = [(2 * self.num_units) // self.num_k_splits, (4 *
                    self.num_units) // self.num_n_splits]
                w = self.add_weight(shape=w_shape, initializer='uniform',
                        dtype=tf.float32, name=f'w_{i}')
                self.ws.append(w)
        super().build(input_state)

    def call(self, x, states):
        def SetDevices(xs):
            new_xs = []
            assert len(xs) <= len(self.devices)
            for x, dev in zip(xs, self.devices):
                if not x.device:
                    with tf.device(dev):
                        new_xs.append(tf.identity(x))
                else:
                    assert x.device == dev
                    new_xs.append(x)
            return new_xs

        assert len(states) == self.num_gpus + self.num_n_splits
        hs = SetDevices(states[:self.num_gpus])
        cs = SetDevices(states[self.num_gpus:self.num_gpus+self.num_n_splits])

        # Replicate tensors stored by k-axis gpus to n-axis gpus
        def Replicate(xs):
            assert len(xs) == self.num_k_splits
            x_copies = []
            for i, x in enumerate(xs):
                for j in range(self.num_n_splits):
                    dev_idx = i*self.num_n_splits+j
                    with tf.device(self.devices[dev_idx]):
                        x_copies.append(tf.identity(x))
            assert len(x_copies) == self.num_gpus
            return x_copies

        assert x.shape.as_list() == [self.batch_size, self.num_units]
        xs = Replicate(tf.split(x, self.num_k_splits, axis=1))

        # Concatenate x and h
        xhs = []
        assert len(xs) == len(hs)
        for x, h in zip(xs, hs):
            assert x.device == h.device
            with tf.device(h.device):
                xhs.append(tf.concat([x, h], axis=1))
        assert len(xhs) == self.num_gpus
        assert all(x.device == device for x, device in zip(xhs, self.devices))

        # GEMM
        xhs = [xhs[i::self.num_n_splits] for i in range(self.num_n_splits)]
        ws_lst = [self.ws[i::self.num_n_splits] for i in range(self.num_n_splits)]
        ifgos = [ParallelGEMM(xs, ws) for xs, ws in zip(xhs, ws_lst)]
        assert len(ifgos) == self.num_n_splits
        assert all(ifgo.device == device for ifgo, device in zip(ifgos,
            self.devices))

        # Apply activations
        assert len(ifgos) == len(cs)
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
            #num_concats = self.num_n_splits // self.num_k_splits
            #new_hs_split = [new_hs_split[i:i+num_concats] for i in range(0,
            #    len(new_hs_split), num_concats)]
            #new_hs = []
            #for d, hs in zip(self.devices[::self.num_n_splits], new_hs_split):
            #    with tf.device(d):
            #        new_hs.append(tf.concat(hs, axis=1))
            #assert [h.device for h in new_hs] == [d.to_string() for d in
            #        self.devices[::self.num_n_splits]]

            # TODO: Handling just this special case for now.
            assert self.num_k_splits == self.num_n_splits // 2

            num_hs = len(new_hs_split)
            num_devices = len(self.devices)
            num_hs_by_2 = num_hs // 2
            num_devices_by_2 = num_devices // 2
            assert num_hs == self.num_n_splits
            assert all(h.device == d for h, d in zip(new_hs_split,
                self.devices))

            split_1 = AllConcatRing(new_hs_split[:num_hs_by_2], axis=1)
            split_2 = AllConcatRing(new_hs_split[num_hs_by_2:], axis=1,
                    devices=self.devices[num_devices_by_2:num_devices_by_2+num_hs_by_2])

            new_hs = split_1[:]
            for h, dev in zip(split_1, self.devices[num_hs_by_2:]):
                with tf.device(dev):
                    new_hs.append(tf.identity(h))
            new_hs += split_2
            for h, dev in zip(split_2, self.devices[-num_hs_by_2:]):
                with tf.device(dev):
                    new_hs.append(tf.identity(h))

        elif self.num_n_splits < self.num_k_splits:
            new_hs = []
            num_splits = self.num_k_splits // self.num_n_splits
            for h in new_hs_split:
                with tf.device(h.device):
                    new_hs += tf.split(h, num_splits, axis=1)

        else:
            new_hs = new_hs_split
            full_new_hs = []
            for i, h in enumerate(new_hs):
                i *= self.num_n_splits
                for j in range(i, i+self.num_k_splits):
                    with tf.device(self.devices[j]):
                        full_new_hs.append(tf.identity(h))
            new_hs = full_new_hs

        assert [h.device for h in new_hs] == [d for d in self.devices]
        assert [c.device for c in new_cs] == [d for d in
                self.devices[:self.num_n_splits]]

        with tf.device(self.devices[0]):
            x = tf.concat(new_hs[::self.num_n_splits], axis=1)
            assert x.shape.as_list() == [self.batch_size, self.num_units]
        return x, new_hs + new_cs


def main():
    parser = argparse.ArgumentParser(formatter_class =
            argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocab', type=str, help="Source vocab data file.")
    parser.add_argument('--text', type=str, help="Source text data file.")
    parser.add_argument('--seq_len', type=int, required=False, default=256,
            help='Maximum sequence length')

    trainer = common.Trainer(parser)
    args = trainer.args
    params = Params(args.batch_size, args.seq_len)
    num_gpus = trainer.num_gpus
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
    vocab_size = int(math.ceil(vocab_size / num_gpus)) * int(num_gpus)
    print("Vocab size: %d" % vocab_size)

    # RNN
    if num_gpus == 8:
        num_k_splits = 2
        num_n_splits = 4
    elif num_gpus == 16:
        num_k_splits = 4
        num_n_splits = 4
    elif num_gpus == 32:
        num_k_splits = 4
        num_n_splits = 8
    else:
        assert False
    cells = [LSTMCell(params.batch_size, params.num_units, devices,
        num_k_splits, num_n_splits),
            LSTMCell(params.batch_size, params.num_units, devices, num_k_splits,
                num_n_splits)]
    rnn = keras.layers.RNN(cells, return_sequences=True, return_state=False)

    # Initial states
    hs0, hs1 = [], []
    for device in devices:
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
    states = [hs0 + cs0, hs1 + cs1]

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
            params.num_units],
            partitioner=tf.fixed_size_partitioner(num_gpus, axis=0))
    embed = tf.nn.embedding_lookup(embedding_weights, inputs)
    xs = rnn(embed, initial_state=states)

    # Final dense layer
    ys = []
    for dev in devices:
        with tf.device(dev):
            ys.append(keras.layers.Dense(vocab_size // num_gpus, use_bias=False)(xs))
    with tf.device(devices[0]):
        y = tf.concat(ys, axis=2)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, y,
                reduction=tf.losses.Reduction.MEAN)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads = opt.compute_gradients(loss, colocate_gradients_with_ops=True)
    assert all(g is not None for g in grads)
    grads = opt.apply_gradients(grads)

    # Train
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    config = tf.ConfigProto(#log_device_placement=True,
            allow_soft_placement=True)
    trainer.train(tf.global_variables_initializer(), loss, [grads], dataset,
            config=config, run_options=run_options)


if __name__ == '__main__':
    main()

