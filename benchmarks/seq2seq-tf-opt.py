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

def Devices(lst):
    return [x.device for x in lst]

def FlattenList(l):
   return [item for sublist in l for item in sublist]

class Params():
    def __init__(self):
        self.num_units = 1024
        self.max_seq_len = 256
        self.num_layers = 4

def AllConcatRing(xs, axis, devices=None):
  devices = devices or [x.device for x in xs]
  n = len(xs)
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
    def __init__(self, num_units, layer, num_layers, devices, num_k_splits,
            num_n_splits, num_gpus, attention=False, **kwargs):
        self.num_gpus = num_gpus
        self.num_units = num_units
        self.devices = devices
        self.layer = layer
        self.num_layers = num_layers
        self.attention = attention
        self.num_k_splits = num_k_splits
        self.num_n_splits = num_n_splits
        assert len(self.devices) == self.num_gpus

        h_state_size = [num_units // self.num_k_splits] * self.num_gpus
        c_state_size = [num_units // self.num_n_splits] * self.num_n_splits
        self.state_size = h_state_size + c_state_size

        if attention and layer == 0:
            self.state_size += h_state_size
        self.output_size = num_units
        if attention and layer != num_layers - 1:
            self.output_size += num_units

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.ws = []
        for d in self.devices:
            with tf.device(d):
                w_shape = [((2 + self.attention) * self.num_units) //
                        self.num_k_splits, (4 * self.num_units) //
                        self.num_n_splits]
                w = self.add_weight(shape=w_shape, initializer='uniform',
                        dtype=tf.float32, name=f'w_{d.device_index}')
                self.ws.append(w)
        super().build(input_shape)

    def call(self, x, states, constants):
        def SetDevices(xs):
            new_xs = []
            for x, dev in zip(xs, self.devices):
                if not x.device:
                    with tf.device(dev):
                        new_xs.append(tf.identity(x))
                else:
                    assert x.device == dev.to_string()
                    new_xs.append(x)
            return new_xs

        assert len(states) >= self.num_gpus + self.num_n_splits
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

        if self.attention:
            # For layer 0, context is computed and stored in states
            # For other layers, context is passed from previous layer, by
            # concatenating with 'x'
            if self.layer == 0:
                assert len(states) == 2*self.num_gpus + self.num_n_splits
                assert x.shape[-1] == self.num_units
                xs = tf.split(x, self.num_k_splits, axis=1)
                contexts = SetDevices(states[-self.num_gpus:])
            else:
                assert len(states) == self.num_gpus + self.num_n_splits
                assert x.shape[-1] == 2 * self.num_units
                splits = tf.split(x, 2 * self.num_k_splits, axis=1)
                xs = splits[:self.num_k_splits]
                contexts = Replicate(splits[self.num_k_splits:])
        else:
            xs = tf.split(x, self.num_k_splits, axis=1)

        def Attention(hs):
            attn_num_k_splits, attn_num_n_splits = 4, 1
            assert len(constants) == attn_num_k_splits + self.num_k_splits
            enc_attn = constants[:attn_num_k_splits]
            enc_out = constants[attn_num_k_splits:]
            assert len(enc_attn) == len(hs)
            assert len(enc_out) == self.num_k_splits

            score = ParallelGEMM(enc_attn, hs, 'ble,be->bl')
            score = tf.nn.softmax(score, axis=-1)
            contexts = []
            for dev, y in zip(self.devices[::self.num_n_splits], enc_out):
                with tf.device(dev):
                    contexts.append(tf.einsum('bl,ble->be', score, y))
            return contexts

        xs = Replicate(xs)
        concat_lst = [hs, xs]
        if self.attention:
            concat_lst.append(contexts)
        assert all(len(x) == self.num_gpus for x in concat_lst)

        # Concatenate x and h
        xhs = []
        for tsrs in zip(*concat_lst):
            assert all(t.device == tsrs[0].device for t in tsrs)
            with tf.device(tsrs[0].device):
                xhs.append(tf.concat(tsrs, axis=1))
        assert len(xhs) == self.num_gpus
        assert all(x.device == device.to_string() for x, device in zip(xhs,
            self.devices))

        # GEMM
        xhs = [xhs[i::self.num_n_splits] for i in range(self.num_n_splits)]
        ws_lst = [self.ws[i::self.num_n_splits] for i in range(self.num_n_splits)]
        ifgos = [ParallelGEMM(xs, ws) for xs, ws in zip(xhs, ws_lst)]
        assert len(ifgos) == self.num_n_splits
        assert all(ifgo.device == device.to_string() for ifgo, device in
                zip(ifgos, self.devices))

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
            #tmp_hs = [new_hs_split[i:i+num_concats] for i in range(0,
            #    len(new_hs_split), num_concats)]
            #new_hs = []
            #for d, hs in zip(self.devices[::self.num_n_splits], tmp_hs):
            #    with tf.device(d):
            #        new_hs.append(tf.concat(hs, axis=1))

            # TODO: Handling just this special case for now.
            assert len(new_hs_split) == 4
            assert all(h.device == dev.to_string() for h, dev in zip(new_hs_split,
                self.devices[:4]))
            split_1 = AllConcat(new_hs_split[:2], axis=1)
            split_2 = AllConcat(new_hs_split[2:], axis=1,
                    devices=self.devices[4:6])
            new_hs = split_1[:]
            for h, dev in zip(split_1, self.devices[2:]):
                with tf.device(dev):
                    new_hs.append(tf.identity(h))
            new_hs += split_2
            for h, dev in zip(split_2, self.devices[6:]):
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

        assert [h.device for h in new_hs] == [d.to_string() for d in
                self.devices]
        assert [c.device for c in new_cs] == [d.to_string() for d in
                self.devices[:self.num_n_splits]]

        with tf.device(self.devices[0]):
            x = tf.concat(new_hs[::self.num_n_splits], axis=1)
            assert x.shape[-1] == self.num_units

        # For layer 0, compute context for next timestep and add it to the
        # states. Except for the last layer, all other layers concatenate
        # 'context' with 'x' and  pass it to next layer.
        states = new_hs + new_cs
        if self.attention:
            if self.layer == 0:
                assert len(new_hs_split) == self.num_n_splits
                next_context = Replicate(Attention(new_hs_split))
                assert len(next_context) == self.num_gpus
                states += next_context

            if self.layer != self.num_layers - 1:
                with tf.device(self.devices[0]):
                    concat_lst = [x] +  contexts[::self.num_n_splits]
                    x = tf.concat(concat_lst, axis=1)
                    assert x.shape[-1] == 2 * self.num_units

        return x, states

class RNN(keras.layers.RNN):
    def build(self, input_shapes):
        assert self.constants_spec is not None
        constants_shape = [spec.shape for spec in self.constants_spec]
        input_shapes = [input_shapes] + constants_shape
        super().build(input_shapes)

class Model():
    def __init__(self, args, params, src_vocab_size, tgt_vocab_size, devices):
        num_gpus = self.num_gpus = 8
        self.devices = devices
        self.num_units = params.num_units
        self.num_layers = params.num_layers
        num_k_splits = self.num_k_splits = 2
        num_n_splits = self.num_n_splits = 4

        def get_device_fn():
            i = 0
            def fn(_):
                nonlocal i
                j = i
                i = (i + 1) % len(devices)
                return devices[j]
            return fn

        # Encoder
        with tf.device(get_device_fn()):
            self.enc_weights = tf.get_variable('encoder_weights',
                    shape=[src_vocab_size, params.num_units],
                    partitioner=tf.fixed_size_partitioner(num_gpus, axis=0))

        # Encoder RNN
        cells = [LSTMCell(params.num_units, i, params.num_layers, devices,
            num_k_splits, num_n_splits, num_gpus) for i in range(4)]
        self.enc_rnn = RNN(cells, return_sequences=True, return_state=True)

        # Encoder attention
        self.enc_attn = lambda x: ParallelDense(x, [params.num_units,
            params.num_units], 2, 4, devices, name='enc_attn')

        # Decoder
        with tf.device(get_device_fn()):
            self.dec_weights = tf.get_variable('decoder_weights',
                    shape=[tgt_vocab_size, params.num_units],
                    partitioner=tf.fixed_size_partitioner(num_gpus, axis=0))

        # Decoder RNN
        cells = [LSTMCell(params.num_units, i, params.num_layers, devices,
            num_k_splits, num_n_splits, num_gpus, True) for i in range(4)]
        self.dec_rnn = RNN(cells, return_sequences=True, return_state=False)

        # Final projection
        self.proj = lambda x: ParallelDense(x, [params.num_units,
            tgt_vocab_size], 1, 8, devices, name='final_proj', concat=True)

    def __call__(self, enc_inp, dec_inp):
        batch_size = enc_inp.shape[0]

        # Initial encoder states
        enc_states = []
        for _ in range(self.num_layers):
            for device in self.devices:
                with tf.device(device):
                    enc_states.append(tf.zeros([batch_size, self.num_units //
                        self.num_k_splits], dtype=tf.float32))
            for device in self.devices[:self.num_n_splits]:
                with tf.device(device):
                    enc_states.append(tf.zeros([batch_size, self.num_units //
                        self.num_n_splits], dtype=tf.float32))

        # Encoder
        x = tf.nn.embedding_lookup(self.enc_weights, enc_inp)
        enc_out, *enc_states = self.enc_rnn(x, initial_state=enc_states,
                constants=tf.zeros(1))
        enc_out = tf.split(enc_out, 2, -1)
        enc_attn = self.enc_attn(enc_out)
        constants = enc_attn + enc_out

        # Decoder
        x = tf.nn.embedding_lookup(self.dec_weights, dec_inp)
        contexts = []
        for dev in self.devices:
            with tf.device(dev):
                contexts.append(tf.zeros([batch_size, self.num_units //
                    self.num_k_splits], dtype=tf.float32))
        cnt = self.num_gpus + self.num_n_splits
        dec_states = enc_states[:cnt] + contexts + enc_states[cnt:]
        dec_out = self.dec_rnn(x, initial_state=dec_states, constants=constants)
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

    model = Model(args, params, src_vocab_size, tgt_vocab_size, devices)
    y = model(enc_inputs, dec_inputs)
    assert y.shape.as_list() == [args.batch, params.max_seq_len, tgt_vocab_size]

    # Loss
    with tf.device(devices[0]):
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
            allow_soft_placement=False)
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

