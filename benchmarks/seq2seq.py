import os
import sys
import copy
import itertools
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf


CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }


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


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    return data


def create_lookup_tables(text):
    # make a list of unique words
    vocab = set(text.split())

    # (1)
    # starts with the special tokens
    vocab_to_int = copy.copy(CODES)

    # the index (v_i) will starts from 4 (the 2nd arg in enumerate() specifies the starting index)
    # since vocab_to_int already contains special tokens
    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    # (2)
    #int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int#, int_to_vocab


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
        1st, 2nd args: raw string text to be converted
        3rd, 4th args: lookup tables for 1st and 2nd args respectively
    
        return: A tuple of lists (source_id_text, target_id_text) converted
    """
    # empty list of converted sentences
    source_text_id = []
    target_text_id = []
    
    # make a list of sentences (extraction)
    source_sentences = source_text.split("\n")
    target_sentences = target_text.split("\n")
    
    max_source_sentence_length = max([len(sentence.split(" ")) for sentence in source_sentences])
    max_target_sentence_length = max([len(sentence.split(" ")) for sentence in target_sentences])
    
    # iterating through each sentences (# of sentences in source&target is the same)
    for i in range(len(source_sentences)):
        # extract sentences one by one
        source_sentence = source_sentences[i]
        target_sentence = target_sentences[i]
        
        # make a list of tokens/words (extraction) from the chosen sentence
        source_tokens = source_sentence.split(" ")
        target_tokens = target_sentence.split(" ")
        
        # empty list of converted words to index in the chosen sentence
        source_token_id = []
        target_token_id = []
        
        for index, token in enumerate(source_tokens):
            if (token != ""):
                source_token_id.append(source_vocab_to_int[token])
        
        for index, token in enumerate(target_tokens):
            if (token != ""):
                target_token_id.append(target_vocab_to_int[token])
                
        # put <EOS> token at the end of the chosen target sentence
        # this token suggests when to stop creating a sequence
        target_token_id.append(target_vocab_to_int['<EOS>'])
            
        # add each converted sentences in the final list
        source_text_id.append(source_token_id)
        target_text_id.append(target_token_id)
    
    return source_text_id, target_text_id


def preprocess(source_path, target_path):
    # Preprocess
    
    # load original data (English, French)
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    # to the lower case
    source_text = source_text.lower()
    target_text = target_text.lower()

    # create lookup tables for English and French data
    source_vocab_to_int = create_lookup_tables(source_text)
    target_vocab_to_int = create_lookup_tables(target_text)

    # create list of sentences whose words are represented in index
    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    return source_text, target_text, len(source_vocab_to_int), len(target_vocab_to_int), source_vocab_to_int['<PAD>'], target_vocab_to_int['<PAD>'], target_vocab_to_int['<GO>']


def enc_dec_model_inputs(batch_size):
    inputs = tf.placeholder(tf.int32, [batch_size, None], name='input')
    targets = tf.placeholder(tf.int32, [batch_size, None], name='targets') 
    
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)    
    
    return inputs, targets, target_sequence_length, max_target_len


#def hyperparam_inputs():
#    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
#    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#    
#    return lr_rate, keep_prob


def process_decoder_input(target_data, go_id, batch_size):
    """
    Preprocess target data for encoding
    :return: Preprocessed target data
    """
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1)
    
    return after_concat


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_vocab_size, 
                   encoding_embedding_size):
    """
    :return: tuple (RNN output, RNN state)
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
                                             vocab_size=source_vocab_size, 
                                             embed_dim=encoding_embedding_size)
    
    #stacked_cells = tf.keras.layers.StackedRNNCells([tf.contrib.rnn.DropoutWrapper(tf.keras.layers.LSTMCell(rnn_size),
    #    keep_prob) for _ in range(num_layers)])
    stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size),
        keep_prob) for _ in range(num_layers)])
    
    outputs, state = tf.nn.dynamic_rnn(stacked_cells, 
                                       embed, 
                                       dtype=tf.float32)
    return outputs, state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    """
    Create a training process in decoding layer 
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, 
                                               target_sequence_length)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_summary_length)
    return outputs


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length, rnn_size,
                   num_layers, target_vocab_size, batch_size, keep_prob,
                   decoding_embedding_size):
    """
    Create decoding layer
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    #cells = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(rnn_size)
    #    for _ in range(num_layers)])
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _
        in range(num_layers)])
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state, 
                                            cells, 
                                            dec_embed_input, 
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)

    return train_output


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length, max_target_sentence_length,
                  max_target_sequence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, go_id):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size, 
                                             enc_embedding_size)
    
    dec_input = process_decoder_input(target_data, 
                                      go_id, 
                                      batch_size)
    
    train_logits = decoding_layer(dec_input,
                                  enc_states, 
                                  target_sequence_length, 
                                  max_target_sentence_length,
                                  rnn_size,
                                  num_layers,
                                  target_vocab_size,
                                  batch_size,
                                  keep_prob,
                                  dec_embedding_size)
    
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    
    # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
    # - Returns a mask tensor representing the first N positions of each cell.
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    
    # Loss function - weighted softmax cross entropy
    #print_op = tf.print({0 : tf.shape(training_logits), 1 :
    #    tf.shape(target_data)}, output_stream = sys.stdout)
    #with tf.control_dependencies([print_op]):
    #    cost = tf.contrib.seq2seq.sequence_loss(training_logits, target_data, masks,
    #            average_across_timesteps = True, average_across_batch = False)
    cost = tf.contrib.seq2seq.sequence_loss(training_logits, target_data, masks,
            average_across_timesteps = True, average_across_batch = True)
        
    return cost


def pad_sentence_batch(sentence_batch, pad_int, pad_size):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    return [sentence + [pad_int] * (pad_size - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_size = max([len(sentence) for sentence in
            itertools.chain(sources_batch, targets_batch)])
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch,
            source_pad_int, pad_size))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch,
            target_pad_int, pad_size))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        assert len(pad_source_lengths) == len(pad_targets_batch)
        for x, y in zip(pad_source_lengths, pad_targets_lengths):
            assert(x == y)

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


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
    parser.add_argument('src', type=str, help="Source language data file.")
    parser.add_argument('tgt', type=str, help="Target language data file.")
    args = vars(parser.parse_args())

    display_step = 10
    
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
    
    source_path = args['src']
    target_path = args['tgt']

    print("Start processing data...")
    source_int_text, target_int_text, source_vocab_to_int_len, target_vocab_to_int_len, src_pad_id, tgt_pad_id, go_id = preprocess(source_path, target_path)
    max_target_sentence_length = max([len(sentence) for sentence in
        itertools.chain(source_int_text, target_int_text)])
    print("End processing data...")

    print("Source vocab length: " + str(source_vocab_to_int_len))
    print("Target vocab length: " + str(target_vocab_to_int_len))
    
    input_data, targets, target_sequence_length, max_target_sequence_length = enc_dec_model_inputs(batch_size)
    #lr, keep_prob = hyperparam_inputs()
    
    split_params = {'input_data' : input_data,
                    'target_data' : targets,
                    'target_sequence_length' : target_sequence_length}

    unsplit_params = {'keep_prob' : keep_prob,
                      'batch_size' : int(batch_size / num_gpus),
                      'max_target_sentence_length' : max_target_sentence_length,
                      'max_target_sequence_length' : max_target_sequence_length,
                      'source_vocab_size' : source_vocab_to_int_len,
                      'target_vocab_size' : target_vocab_to_int_len,
                      'enc_embedding_size' : encoding_embedding_size,
                      'dec_embedding_size' : decoding_embedding_size,
                      'rnn_size' : rnn_size,
                      'num_layers' : num_layers,
                      'go_id' : go_id}

    cost = make_data_parallel(seq2seq_model, num_gpus, split_params, unsplit_params)
    cost = tf.reduce_mean(cost)
    
    with tf.name_scope("optimization"):
        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)
    
        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost,
                colocate_gradients_with_ops = True)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
    
        for epoch_i in range(epochs):
            for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                    get_batches(source_int_text, target_int_text, batch_size,
                                src_pad_id, tgt_pad_id)):
    
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: source_batch,
                     targets: target_batch,
                     target_sequence_length: targets_lengths})
    
    
                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3} Batch {:>4}/{} - Loss: '
                          .format(epoch_i, batch_i, len(source_int_text) // batch_size))
                    print(loss)
    
if __name__ == "__main__":
    main()
