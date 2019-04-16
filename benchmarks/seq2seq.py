import os
import time
import sys
import copy
import itertools
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf

from dataloader import TextDataLoader


def get_gpu_device(idx):
    return tf.device(tf.DeviceSpec(device_type = "GPU", device_index = idx))


def make_data_parallel(fn, num_gpus, split_args, unsplit_args):
    in_splits = {}
    for k, v in split_args.items():
        in_splits[k] = tf.split(v, num_gpus, axis = 0)

    unsplit_args['batch_size'] //= int(num_gpus)

    out_split = []
    for i in range(num_gpus):
        with get_gpu_device(i):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()},
                    **unsplit_args))

    return tf.stack(out_split, axis=0)


class LSTMCell(tf.contrib.rnn.LayerRNNCell):
    def __init__(self, num_units, num_gpus,
                 initializer=None, forget_bias=1.0, activation=None, reuse=None,
                 name=None, dtype=None, **kwargs):
        super(LSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype,
                **kwargs)

        assert num_units % num_gpus == 0

        self._num_units = num_units
        self._num_units_per_gpu = num_units / num_gpus
        self._initializer = tf.keras.initializers.get(initializer)
        self._num_proj = None
        self._forget_bias = forget_bias
        self._state_is_tuple = True
        if activation:
          self._activation = tf.keras.activations.get(activation)
        else:
          self._activation = tf.math.tanh

        self._state_size = \
                tf.nn.rnn_cell.LSTMStateTuple(self._num_units_per_gpu,
                        self._num_units_per_gpu)
        self._output_size = self._num_units_per_gpu

    def build(self, inputs_shape):
      if inputs_shape[-1] is None:
        raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                         % str(inputs_shape))

      _BIAS_VARIABLE_NAME = "bias"
      _WEIGHTS_VARIABLE_NAME = "kernel"

      input_depth = inputs_shape[-1]
      h_depth = self._num_units if self._num_proj is None else self._num_proj
      self._kernel = self.add_variable(
          _WEIGHTS_VARIABLE_NAME,
          shape=[input_depth + h_depth, 4 * self._num_units_per_gpu],
          initializer=self._initializer,
          partitioner=None)
      if self.dtype is None:
        initializer = tf.keras.initializers.Zeros
      else:
        initializer = tf.keras.initializers.Zeros(dtype=self.dtype)
      self._bias = self.add_variable(
          _BIAS_VARIABLE_NAME,
          shape=[4 * self._num_units_per_gpu],
          initializer=initializer)

      if self._num_proj is not None:
        self._proj_kernel = self.add_variable(
            "projection/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self._num_units_per_gpu, self._num_proj],
            initializer=self._initializer,
            partitioner=None)

      self.built = True

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        num_proj = self._num_units_per_gpu if self._num_proj is None else self._num_proj
        sigmoid = tf.math.sigmoid

        assert isinstance(state, tf.nn.rnn_cell.LSTMStateTuple)

        (c_prev, m_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
          raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = tf.matmul(
            tf.concat([inputs, m_prev], 1), self._kernel)
        lstm_matrix = tf.nn.bias_add(lstm_matrix, self._bias)

        i, j, f, o = tf.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        c = (tf.sigmoid(f + self._forget_bias) * c_prev + tf.sigmoid(i) *
             self._activation(j))

        m = tf.sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
          m = tf.matmul(m, self._proj_kernel)

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, m)
        return m, new_state


class ConcatWrapper(tf.contrib.rnn.LayerRNNCell):
    def __init__(self, cells):
        super(ConcatWrapper, self).__init__()
        self._cells = cells

        _output_size = 0
        _state_size_c = [0] * len(cells[0].state_size)
        _state_size_h = [0] * len(cells[0].state_size)
        for cell in cells:
            for i, (_c, _h) in enumerate(cell.state_size):
                _state_size_c[i] += _c
                _state_size_h[i] += _h

            _output_size += cell.output_size

        self._state_size = tuple(tf.nn.rnn_cell.LSTMStateTuple(_c, _h) for _c,
                _h in zip(_state_size_c, _state_size_h))
        self._output_size = _output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inp, st, scope = None):
        new_states_c = {}
        new_states_h = {}
        final_states = []
        final_output = []

        for i, cell in enumerate(self._cells):
            with get_gpu_device(i):
                curr_states = []
                # Iterate over each layer state of multicell
                for state in st:
                    # Get i-th c-state, and h-state
                    curr_states.append(tf.nn.rnn_cell.LSTMStateTuple(state.c[i],
                        state.h))
                curr_states = tuple(curr_states)

                # Call the multicell
                output, states = cell(inp, curr_states, scope = scope)

                final_output.append(output)

                # Iterate over each layer cell output
                for l, state in enumerate(states):
                    # Append state of l-th cell to l-th new_states
                    try:
                        new_states_c[l].append(state.c)
                        new_states_h[l].append(state.h)
                    except KeyError:
                        new_states_c[l] = [state.c]
                        new_states_h[l] = [state.h]

        # Concatenate the outputs of different devices, and pack the outputs of
        # different layer
        final_output = tf.concat(final_output, 1)
        for l, h in new_states_h.items():
            new_state_h = tf.concat(h, 1)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_states_c[l],
                    new_state_h)
            final_states.append(new_state)
        final_states = tuple(final_states)

        return (final_output, final_states)


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
        source_vocab_size, encoding_embedding_size):
    with tf.variable_scope("encode"):
        embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
                                                 vocab_size=source_vocab_size, 
                                                 embed_dim=encoding_embedding_size)
        
        stacked_cells = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size,
                    name = 'encoder_lstm', reuse = tf.AUTO_REUSE), keep_prob) for _ in
                    range(num_layers)])
        
        outputs, state = tf.nn.dynamic_rnn(stacked_cells, embed, dtype=tf.float32)
        return outputs, state


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, rnn_size,
                   num_layers, target_vocab_size, batch_size, keep_prob,
                   decoding_embedding_size):
    with tf.variable_scope("decode"):
        dec_embeddings = tf.get_variable('dec_embeddings', [target_vocab_size,
            decoding_embedding_size])
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
        
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(rnn_size, name = 'decoder_lstm', reuse =
                tf.AUTO_REUSE), keep_prob) for _ in range(num_layers)])
    
        # for only input layer
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, 
                                                   target_sequence_length)
        
        output_layer = tf.layers.Dense(target_vocab_size)
        decoder = tf.contrib.seq2seq.BasicDecoder(cells, helper, encoder_state,
                output_layer)

        # unrolling the decoder layer
        train_output, _, final_seq_len = tf.contrib.seq2seq.dynamic_decode(decoder,
                impute_finished=True)

        return train_output, final_seq_len


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers):
    with tf.variable_scope('seq2seq'):
        _, enc_states = encoding_layer(input_data, 
                                                 rnn_size, 
                                                 num_layers, 
                                                 keep_prob, 
                                                 source_vocab_size, 
                                                 enc_embedding_size)
        
        max_seq_len = tf.reduce_max(target_sequence_length)
        target_data = tf.slice(target_data, [0, 0], [batch_size, max_seq_len])
        train_logits, final_seq_len = decoding_layer(target_data,
                                      enc_states, 
                                      target_sequence_length, 
                                      rnn_size,
                                      num_layers,
                                      target_vocab_size,
                                      batch_size,
                                      keep_prob,
                                      dec_embedding_size)
        
        training_logits = tf.identity(train_logits.rnn_output, name='logits')
        
        masks = tf.sequence_mask(final_seq_len, dtype=tf.float32, name='masks')

        # Loss function - weighted softmax cross entropy
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, target_data, masks,
                average_across_timesteps = True, average_across_batch = True)
            
        return cost


def ConcatenatedLSTM(rnn_size, keep_prob, num_gpus, num_layers, name):
    cells = []
    for i in range(num_gpus):
        with get_gpu_device(i):
            cells.append(tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(LSTMCell(rnn_size, num_gpus, 
                    name = name + str(i), reuse = False), keep_prob) for _ in
                    range(num_layers)]))

    cell = ConcatWrapper(cells)
    return cell


def ConcatenateStates(states):
    concat_states = []
    for state in states:
        c_state = tf.concat(state.c, axis = 1)
        h_state = state.h
        concat_states.append(tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state))

    return tuple(concat_states)


def seq2seq_opt4(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers):
    num_gpus = 4

    with tf.variable_scope('seq2seq', reuse = False):
        # Encoder
        with tf.variable_scope('encode', reuse = False):
            with get_gpu_device(0):
                enc_embed = tf.contrib.layers.embed_sequence(input_data,
                        vocab_size=source_vocab_size,
                        embed_dim=enc_embedding_size)

            initial_c_states = []
            for i in range(num_gpus):
                with get_gpu_device(i):
                    initial_c_state = tf.zeros([batch_size, rnn_size /
                        num_gpus])
                    initial_c_states.append(initial_c_state)

            initial_h_state = tf.zeros([batch_size, rnn_size])
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_c_states,
                    initial_h_state)
            initial_state = (initial_state,) * num_layers

            cell = ConcatenatedLSTM(rnn_size, keep_prob, num_gpus, num_layers,
                    'encoder_lstm')
            _, enc_states = tf.nn.dynamic_rnn(cell, enc_embed, initial_state =
                    initial_state, dtype=tf.float32)

            with get_gpu_device(0):
                enc_states = ConcatenateStates(enc_states)

        # Decoder
        with tf.variable_scope('decode', reuse = False):
            with get_gpu_device(0):
                dec_embed = tf.get_variable('dec_embed',
                        [target_vocab_size, dec_embedding_size])
                dec_embed_input = tf.nn.embedding_lookup(dec_embed, target_data)

            cell = ConcatenatedLSTM(rnn_size, keep_prob, num_gpus, num_layers,
                    'decoder_lstm')

            # for only input layer
            output_layer = tf.layers.Dense(target_vocab_size)
            helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, 
                                                       target_sequence_length)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, enc_states,
                    output_layer)
            train_output, _, final_seq_len = tf.contrib.seq2seq.dynamic_decode(decoder,
                    impute_finished=True)

        # Loss
        with tf.variable_scope('loss', reuse = False):
            training_logits = tf.identity(train_output.rnn_output, name='logits')
            masks = tf.sequence_mask(final_seq_len, dtype=tf.float32, name='masks')

            # Loss function - weighted softmax cross entropy
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, target_data, masks,
                    average_across_timesteps = True, average_across_batch = True)

            return cost


def main():
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, required=False, default=256,
            help="Batch size. (Default: 256)")
    parser.add_argument('-p', '--procs', type=int, required=False, default=4,
            help="No. of processors. (Default: 4)")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
            help="No. of epochs")
    parser.add_argument('-l', '--layers', type=int, required=False,
            default=2, help="Number of RNN layers. (Default: 2)")
    parser.add_argument('-d', '--hidden', type=int, required=False,
            default=1024, help="Size of hidden dimension. (Default: 1024)")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(2)), 
            help="Strategy to use. 0: DataParallel, 1: Optimized. (Default: 0)")
    parser.add_argument('src_vocab', type=str, help="Source vocab data file.")
    parser.add_argument('tgt_vocab', type=str, help="Target vocab data file.")
    parser.add_argument('src_text', type=str, help="Source text data file.")
    parser.add_argument('tgt_text', type=str, help="Target text data file.")
    args = vars(parser.parse_args())

    display_step = 100
    strategy = args['strategy']
    
    epochs = args['epochs']
    batch_size = args['batch']
    
    rnn_size = args['hidden']
    num_layers = args['layers']
    
    encoding_embedding_size = args['hidden']
    decoding_embedding_size = args['hidden']
    
    lr = 0.001
    keep_prob = 0.5

    num_gpus = args['procs']
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(num_gpus))[:-1]
    
    src_vocab = args['src_vocab']
    tgt_vocab = args['tgt_vocab']
    src_text = args['src_text']
    tgt_text = args['tgt_text']

    with open(src_vocab) as f:
        for src_vocab_size, _ in enumerate(f):
            pass
    with open(tgt_vocab) as f:
        for tgt_vocab_size, _ in enumerate(f):
            pass

    print("Source vocab length: " + str(src_vocab_size))
    print("Target vocab length: " + str(tgt_vocab_size))
    
    dataset = TextDataLoader(batch_size, src_vocab, tgt_vocab, src_text,
            tgt_text)
    input_data, targets, target_sequence_length = dataset.next_batch()
    
    split_params = {'input_data' : input_data,
                    'target_data' : targets,
                    'target_sequence_length' : target_sequence_length}

    unsplit_params = {'keep_prob' : keep_prob,
                      'batch_size' : batch_size,
                      'source_vocab_size' : src_vocab_size,
                      'target_vocab_size' : tgt_vocab_size,
                      'enc_embedding_size' : encoding_embedding_size,
                      'dec_embedding_size' : decoding_embedding_size,
                      'rnn_size' : rnn_size,
                      'num_layers' : num_layers}

    if strategy == 0:
        cost = make_data_parallel(seq2seq_model, num_gpus, split_params, unsplit_params)
    else:
        assert(strategy == 1)
        if num_gpus == 4:
            cost = seq2seq_opt4(**split_params, **unsplit_params)
        else:
            assert(num_gpus == 8)

    cost = tf.reduce_mean(cost)
    
    with tf.variable_scope("optimization", reuse=tf.AUTO_REUSE):
        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)
    
        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost,
                colocate_gradients_with_ops = True)
        #gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(gradients)


    tot_time = float(0)
    cnt = 0
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run([tf.global_variables_initializer(),
            tf.initializers.tables_initializer()])
    
        start = time.time()
        for epoch_i in range(epochs):
            step = 0
            dataset.reset_pointer()

            while True:
                try:
                    _, loss = sess.run([train_op, cost])
                    step += 1
                    cnt += 1
                except tf.errors.OutOfRangeError:
                    break

                if step % display_step == 0 and step > 0:
                    print('Epoch {:>3} Batch {:>4} - Loss: '.format(epoch_i,
                        step))
                    print(loss)

        end = time.time()
        tot_time = (end - start)

    samples_per_sec = (batch_size * cnt) / tot_time
    print("Throughout: " + str(samples_per_sec) + " samples / sec")

    
if __name__ == "__main__":
    main()
