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

    out_split = []
    for i in range(num_gpus):
        with get_gpu_device(i):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()},
                    **unsplit_args))

    return tf.stack(out_split, axis=0)


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
        source_vocab_size, encoding_embedding_size):
    with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
        embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
                                                 vocab_size=source_vocab_size, 
                                                 embed_dim=encoding_embedding_size)
        
        stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size),
            keep_prob) for _ in range(num_layers)])
        
        outputs, state = tf.nn.dynamic_rnn(stacked_cells, embed, dtype=tf.float32)
        return outputs, state


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, rnn_size,
                   num_layers, target_vocab_size, batch_size, keep_prob,
                   decoding_embedding_size):
    with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,
            decoding_embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
        
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _
            in range(num_layers)])
    
        output_layer = None #tf.layers.Dense(target_vocab_size)
        cells = tf.contrib.rnn.DropoutWrapper(cells, output_keep_prob =
                keep_prob)
        
        # for only input layer
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, 
                                                   target_sequence_length)
        
        decoder = tf.contrib.seq2seq.BasicDecoder(cells, helper, encoder_state,
                output_layer)

        # unrolling the decoder layer
        train_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                impute_finished=True)

        return train_output


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers):
    enc_outputs, enc_states = encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size, 
                                             enc_embedding_size)
    
    max_seq_len = tf.reduce_max(target_sequence_length)
    target_data = tf.slice(target_data, [0, 0], [batch_size, max_seq_len])
    train_logits = decoding_layer(target_data,
                                  enc_states, 
                                  target_sequence_length, 
                                  rnn_size,
                                  num_layers,
                                  target_vocab_size,
                                  batch_size,
                                  keep_prob,
                                  dec_embedding_size)
    
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    
    masks = tf.sequence_mask(target_sequence_length, maxlen = max_seq_len,
            dtype=tf.float32, name='masks')

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
                      'batch_size' : int(batch_size / num_gpus),
                      'source_vocab_size' : src_vocab_size,
                      'target_vocab_size' : tgt_vocab_size,
                      'enc_embedding_size' : encoding_embedding_size,
                      'dec_embedding_size' : decoding_embedding_size,
                      'rnn_size' : rnn_size,
                      'num_layers' : num_layers}

    cost = make_data_parallel(seq2seq_model, num_gpus, split_params, unsplit_params)
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
