import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf 

import math
import sys 
import time
import os
from argparse import ArgumentParser
from collections import namedtuple

from dataloader import ImageDataLoader


def AssignLayout(ta_axes, mesh_axis):
    layout = []
    for a in ta_axes:
        layout.append((a, mesh_axis))
    return layout


def Transformer(batch_size, src_vocab, tgt_vocab, src_text, tgt_text):
    # Parameters
    nx = 6
    r = 2
    c = 4

    # Dataset
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
    
    # Mesh
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, 'mesh')
    mesh_shape = [('rows', r), ('cols', c)]
    devices = ['gpu:%d' % i for i in range(r*c)]
    row_layout = AssignLayout(['batch'], 'rows')
    col_layout = AssignLayout(['vocab', 'd_ff', 'heads'], 'cols')
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout,
            devices)
    mesh_to_impl = {mesh:mesh_impl}

    # mtf dimensions
    mtf_batch = mtf.Dimension('batch', batch)
    mtf_seq_len = mtf.Dimension('length_dim', seq_len)
    mtf_src_vocab_dim = mtf.Dimension('vocab_dim', src_vocab_dim)
    mtf_tgt_vocab_dim = mtf.Dimension('vocab_dim', tgt_vocab_dim)
    mtf_heads = mtf.Dimension('heads', heads)
    mtf_d_k = mtf.Dimension('d_k', d_model // heads)
    mtf_memory_len_dim = mtf.Dimension('memory_len_dim', seq_len)
    mtf_d_ff = mtf.Dimension('d_ff', d_ff)

    # Layers
    def MultiheadAttention(q, k, v, dropout_rate=0.1):
        d_model = mtf_d_model.size
    
        k = mtf.layers.Dense(k, mtf.Dimension('heads', d_model), use_bias=False,
                name='linear_k')
        k = mtf.replace_dimensions(k, k.shape[-1], [mtf_heads, mtf_d_k])
        k = mtf.replace_dimensions(k, k.shape[0], [mtf_batch, mtf_seq_len])
    
        v = mtf.layers.Dense(v, mtf.Dimension('heads', d_model), use_bias=False,
                name='linear_v')
        v = mtf.replace_dimensions(v, v.shape[-1], [mtf_heads, mtf_d_k])
        v = mtf.replace_dimensions(v, v.shape[0], [mtf_batch, mtf_seq_len])
        v = mtf.rename_dimension(v, mtf_seq_len.name, mtf_memory_len_dim.name)
    
        q = mtf.layers.Dense(q, mtf.Dimension('heads', d_model), use_bias=False,
                name='linear_k')
        q = mtf.replace_dimensions(q, q.shape[-1], [mtf_heads, mtf_d_k])
        q = mtf.replace_dimensions(q, q.shape[0], [mtf_batch, mtf_seq_len])
    
        assert q.shape == k.shape == mtf.Shape([mtf_batch, mtf_seq_len,
            mtf_heads, mtf_d_k])
        k = mtf.rename_dimension(k, mtf_seq_len.name, mtf_memory_len_dim.name)
    
        qk = mtf.einsum([q, k], reduced_dims=[mtf_d_k], name='attention_qk')
        qk /= math.sqrt(mtf_d_k.size)
        assert qk.shape == mtf.Shape([mtf_batch, mtf_seq_len, mtf_heads,
            mtf_memory_len_dim)])
    
        weights = mtf.softmax(qk, mtf_memory_len_dim)
        weights = mtf.dropout(weights, dropout_rate, noise_shape=weights.shape -
                mtf_seq_len, name='weights_dropout')
    
        outputs = mtf.einsum([weights, v], reduced_dims=[mtf_memory_len_dim])
    
        assert outputs.shape == q.shape
        return outputs
    
    
    def FeedFwd(x, dropout_rate=0.1):
        x = mtf.layers.Dense(x, mtf_d_ff, use_bias=False, activation=mtf.relu,
                name='linear_ff1')
        x = mtf.dropout(x, dropout_rate, noise_shape=x.shape - mtf_seq_len,
                name='ff_dropout')
        x = mtf.layers.Dense(x, mtf_d_model, use_bias=False, name='linear_ff2')
        return x
    
    
    def EncoderLayer(x, dropout_rate=0.1):
        norm1 = mtf.layer_norm(x, dim=x.shape[-1], name='enc_norm1')
        att = MultiheadAttention(norm1, norm1, norm1)
        x += mtf.dropout(att, dropout_rate)
    
        norm2 = mtf.layer_norm(x, dim=x.shape[-1], name='enc_norm2')
        ff = FeedFwd(norm2)
        x += mtf.dropout(ff, dropout_rate)
    
        return x
    
    
    def DecoderLayer(x, enc_out, dropout_rate=0.1):
        norm1 = mtf.layer_norm(x, dim=x.shape[-1], name='dec_norm1')
        att = MultiheadAttention(norm1, norm1, norm1)
        x += mtf.dropout(att, dropout_rate)
    
        norm2 = mtf.layer_norm(x, dim=x.shape[-1], name='dec_norm2')
        att = MultiheadAttention(norm2, enc_out, enc_out)
        x += mtf.dropout(att, dropout_rate)
    
        norm3 = mtf.layer_norm(x, dim=x.shape[-1], name='dec_norm3')
        ff = FeedFwd(norm3)
        x += mtf.dropout(ff, dropout_rate)
    
        return x

    # Model - Encoder embedding
    with tf.variable_scope('encoder'):
        embed = mtf.layers.embedding(enc_inputs, mtf_src_vocab_dim, mtf_d_model,
                enc_inputs.dtype, name='enc_embedding')
        seq_len = embed.shape[-1].size

        # Values for positional encoder
        pos = np.array(tuple(range(length_dim))).reshape(-1, 1)
        val = np.power(10000, ((2 * np.array(tuple(range(d_model)))) / d_model))
        pos_enc_values = pos / val
        np.sin(pos_enc_values[:,::2], out=pos_enc_values[:,::2])
        np.cos(pos_enc_values[:,1::2], out=pos_enc_values[:,1::2])
        
        # positional encoder
        pos_enc = mtf.get_variable(mesh, 'pos_enc', shape=mtf.Shape([mtf_seq_len,
            mtf_d_model]), dtype=tf.float32, initializer=pos_enc_values,
            trainable=False)
        x = (embed * math.sqrt(mtf_d_model.size)) + pos_enc

        # Encoder
        for _ in range(nx):
            x = EncoderLayer(x)
        enc_output = mtf.layer_norm(x, name='enc_final_norm')

    # Decoder embedding
    with tf.variable_scope('decoder'):
        embed = mtf.layers.embedding(dec_inputs, mtf_tgt_vocab_dim, mtf_d_model,
                dec_inputs.dtype, name='dec_embedding')

        # positional encoder
        x = (embed * math.sqrt(mtf_d_model.size)) + pos_enc

        # Decoder
        for _ in range(nx):
            x = DecoderLayer(x, enc_output)
        dec_output = mtf.layer_norm(x, name='dec_final_norm')

    # Linear + softmax + cross-entropy
    with tf.variable_scope('loss'):
        out = mtf.layers.Dense(dec_output, mtf_tgt_vocab_dim,
                name='final_projection')
        one_hot_labels = mtf.one_hot(mtf_labels, out.shape[-1])
        out = mtf.layers.softmax_cross_entropy_with_logits(out, one_hot_labels,
                out.shape[-1])
        mtf_loss = mtf.reduce_mean(out)

    with tf.variable_scope('optimize'):
        grads = mtf.gradient([mtf_loss], [v.outputs[0] for v in
            graph.trainable_variables])
        opt = mtf.optimize.SgdOptimizer(learning_rate)
        grad_updates = opt.apply_grads(grads, graph.trainable_variables)

    lowering = mtf.Lowering(graph, meshes)
    tf_loss = lowering.export_to_tf_tensor(mtf_loss)
    tf_grad_updates = [lowering.lowered_operation(op) for op in grad_updates]

    # Initializer
    tf_init_vars = \
            FlattenList([lowering.variables[var].laid_out_tensor.all_slices for
                var in graph.trainable_variables])
    init_op = []
    for v in tf_init_vars:
        with tf.device(v.device):
            init_op.append(v.initializer)

    # Training
    cnt = 0
    with tf.variable_scope('train'):
        with tf.Session() as sess:
            dataset.reset_pointer()
            sess.run(init_op)

            tot_time = float(0)
            start = time.time()
            for epoch in range(num_epochs):
                step = 0

                while True:
                    try:
                        loss_val, *rem = sess.run([tf_loss] + tf_grad_updates)
                        cnt += 1
                        step += 1
                    except tf.errors.OutOfRangeError:
                        break

                    if step % display_step == 0 and step > 0:
                        print("Epoch: " + str(epoch) + "; Loss: " + str(loss_val))

                dataset.reset_pointer()
            end = time.time()
            tot_time += (end - start)

    samples_per_sec = (batch_size * cnt) / tot_time
    print("Throughout: " + str(samples_per_sec) + " samples / sec")


def main():
    parser = ArgumentParser()

    parser.add_argument('-b', '--batch', type=int, required=False, default=256,
            help="Batch size. (Default: 64)")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors. (Default: 8)")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
            help="No. of epochs")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(3)), 
            help="Strategy to use. 0: DataParallel, \
                    1: Optimized for 1080Ti, \
                    2: Optimized for DGX. \
                    (Default: 0) ")
    parser.add_argument('src_vocab', type=str, help="Source vocab data file.")
    parser.add_argument('tgt_vocab', type=str, help="Target vocab data file.")
    parser.add_argument('src_text', type=str, help="Source text data file.")
    parser.add_argument('tgt_text', type=str, help="Target text data file.")
    args = vars(parser.parse_args())

    # Input parameters
    num_gpus = args['procs']
    batch_size = args['batch']
    num_epochs = args['epochs']
    strategy = args['strategy']
    num_classes = 1000
    learning_rate = 0.01
    display_step = 10
    warmup = 10
    src_vocab = args['src_vocab']
    tgt_vocab = args['tgt_vocab']
    src_text = args['src_text']
    tgt_text = args['tgt_text']

    if num_gpus != 8 and strategy > 0:
        raise NotImplementedError('Current implementation only handles 8 GPUs \
                for model parallel strategies.')

    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(num_gpus))[:-1]
            
    Transformer(batch_size, src_vocab, tgt_vocab, src_text, tgt_text)
    

if __name__ == '__main__':
    main()

