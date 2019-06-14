import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf 

import math
import sys 
import time
import os
from argparse import ArgumentParser
from collections import namedtuple

from dataloader import TextDataLoader


def FlattenList(l):
   return [item for sublist in l for item in sublist]


def AssignLayout(ta_axes, mesh_axis):
    layout = []
    for a in ta_axes:
        layout.append((a, mesh_axis))
    return layout


def Transformer(args):
    # Parameters
    r = 2
    c = 4
    #nx = 6
    #max_seq_len = 1024
    #d_model = 1024
    #heads = 8
    #d_ff = heads * 512
    nx = 6
    max_seq_len = 1024
    d_model = 512
    heads = 4
    d_ff = 2048

    # Dataset
    dataset = TextDataLoader(args['batch'], args['src_vocab'],
            args['tgt_vocab'], args['src_text'], args['tgt_text'], max_seq_len =
            max_seq_len)
    src_pad_id = tf.cast(dataset.src_pad_id, tf.int32)
    enc_inputs, dec_inputs, _, _ = dataset.next_batch()

    with open(args['src_vocab']) as f:
        for src_vocab_size, _ in enumerate(f):
            pass
    with open(args['tgt_vocab']) as f:
        for tgt_vocab_size, _ in enumerate(f):
            pass
    print("Source vocab size: %d" % src_vocab_size)
    print("Target vocab size: %d" % tgt_vocab_size)

    # Make the vocabulary size a multiple of mesh size
    src_vocab_size = int(math.ceil(src_vocab_size / c)) * int(c)
    tgt_vocab_size = int(math.ceil(tgt_vocab_size / c)) * int(c)

    # mtf dimensions
    mtf_batch = mtf.Dimension('batch', args['batch'])
    mtf_d_model = mtf.Dimension('d_model', d_model)
    mtf_seq_len = mtf.Dimension('length_dim', max_seq_len)
    mtf_src_vocab_dim = mtf.Dimension('vocab_dim', src_vocab_size)
    mtf_tgt_vocab_dim = mtf.Dimension('vocab_dim', tgt_vocab_size)
    mtf_heads = mtf.Dimension('heads', heads)
    mtf_d_k = mtf.Dimension('d_k', d_model // heads)
    mtf_memory_len_dim = mtf.Dimension('memory_len_dim', max_seq_len)
    mtf_d_ff = mtf.Dimension('d_ff', d_ff)

    # Mesh
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, 'mesh')
    mesh_shape = [('rows', r), ('cols', c)]
    devices = ['gpu:%d' % i for i in range(r*c)]
    row_layout = AssignLayout([mtf_batch.name], 'rows')
    col_layout = AssignLayout([mtf_src_vocab_dim.name, mtf_d_ff.name,
        mtf_heads.name], 'cols')
    layout = row_layout + col_layout
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout,
            devices)
    mesh_to_impl = {mesh:mesh_impl}

    # Layers
    def MultiheadAttention(q, k, mask, dropout_rate=0.5,
            name='multihead_attention'):
        k = mtf.rename_dimension(k, mtf_seq_len.name, mtf_memory_len_dim.name)
        out = mtf.layers.multihead_attention(q, k, mask, mtf_d_k, mtf_heads,
                dropout_rate, name=name)
        return out
    
    
    def EncoderLayer(x, mask, dropout_rate=0.5):
        norm1 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='enc_norm1')
        att = MultiheadAttention(norm1, norm1, mask, name='enc_multihead_att')
        x += mtf.dropout(att, dropout_rate)
    
        norm2 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='enc_norm2')
        ff = mtf.layers.dense_relu_dense(norm2, mtf_d_ff, dropout_rate,
                name='enc_ff')
        x += mtf.dropout(ff, dropout_rate)
    
        return x
    
    
    def DecoderLayer(x, enc_out, enc_mask, dec_mask, dropout_rate=0.5):
        norm1 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm1')
        att = MultiheadAttention(norm1, norm1, dec_mask,
                name='dec_multihead_att1')
        x += mtf.dropout(att, dropout_rate)
    
        norm2 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm2')
        att = MultiheadAttention(norm2, enc_out, enc_mask,
                name='dec_multihead_att2')
        x += mtf.dropout(att, dropout_rate)
    
        norm3 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm3')
        ff = mtf.layers.dense_relu_dense(norm3, mtf_d_ff, dropout_rate,
                name='dec_ff')
        x += mtf.dropout(ff, dropout_rate)
    
        return x


    # Model - Encoder embedding
    with tf.variable_scope('encoder'):
        mtf_enc_inputs = mtf.cast(mtf.import_tf_tensor(mesh, enc_inputs,
            mtf.Shape([mtf_batch, mtf_seq_len])), tf.int32)

        embed = mtf.layers.embedding(mtf_enc_inputs, mtf_src_vocab_dim,
                mtf_d_model, tf.float32, name='enc_embedding')
        assert embed.shape == mtf.Shape([mtf_batch, mtf_seq_len, mtf_d_model])

        # Values for positional encoder
        pos = np.array(tuple(range(max_seq_len))).reshape(-1, 1)
        val = np.power(10000, ((2 * np.array(tuple(range(d_model))))
            / d_model), dtype=float)
        pos_enc_values = np.divide(pos, val, dtype=np.float32)
        np.sin(pos_enc_values[:,::2], out=pos_enc_values[:,::2],
                dtype=np.float32)
        np.cos(pos_enc_values[:,1::2], out=pos_enc_values[:,1::2],
                dtype=np.float32)
        
        # positional encoder
        pos_enc = mtf.get_variable(mesh, 'pos_enc',
                shape=mtf.Shape([mtf_seq_len, mtf_d_model]), dtype=tf.float32,
                initializer=tf.constant_initializer(pos_enc_values),
                trainable=False)
        x = (embed * math.sqrt(mtf_d_model.size)) + pos_enc

        # Mask
        #enc_mask = mtf.cast(mtf.equal(mtf_enc_inputs, src_pad_id),
        #        dtype=tf.float32) * -1e9
        enc_mask = None

        # Encoder
        for i in range(nx):
            with tf.variable_scope('enc_layer_%d' % i):
                x = EncoderLayer(x, enc_mask)
        enc_output = mtf.layers.layer_norm(x, dim=x.shape[-1],
                name='enc_final_norm')
        assert enc_output.shape == mtf.Shape([mtf_batch, mtf_seq_len,
            mtf_d_model])

    # Decoder embedding
    with tf.variable_scope('decoder'):
        mtf_dec_inputs = mtf.cast(mtf.import_tf_tensor(mesh, dec_inputs,
            mtf.Shape([mtf_batch, mtf_seq_len])), tf.int32)

        embed = mtf.layers.embedding(mtf_dec_inputs, mtf_tgt_vocab_dim,
                mtf_d_model, tf.float32, name='dec_embedding')

        # positional encoder
        x = (embed * math.sqrt(mtf_d_model.size)) + pos_enc

        # mask
        dec_mask = None

        # Decoder
        for i in range(nx):
            with tf.variable_scope('dec_layer_%d' % i):
                x = DecoderLayer(x, enc_output, enc_mask, dec_mask)
        dec_output = mtf.layers.layer_norm(x, dim=x.shape[-1],
                name='dec_final_norm')
        assert dec_output.shape == mtf.Shape([mtf_batch, mtf_seq_len,
            mtf_d_model])

    # Linear + softmax + cross-entropy
    with tf.variable_scope('loss'):
        out = mtf.layers.dense(dec_output, mtf_tgt_vocab_dim,
                name='final_projection')
        one_hot_labels = mtf.one_hot(mtf_dec_inputs, out.shape[-1], dtype=out.dtype)
        out = mtf.layers.softmax_cross_entropy_with_logits(out, one_hot_labels,
                out.shape[-1])
        mtf_loss = mtf.reduce_mean(out)

    with tf.variable_scope('optimize'):
        grads = mtf.gradients([mtf_loss], [v.outputs[0] for v in
            graph.trainable_variables])
        lr = 0.01
        opt = mtf.optimize.SgdOptimizer(lr)
        grad_updates = opt.apply_grads(grads, graph.trainable_variables)

    print('Lowering mtf ops...', flush=True)
    lowering = mtf.Lowering(graph, mesh_to_impl)
    print('Finished lowering.', flush=True)
    tf_loss = lowering.export_to_tf_tensor(mtf_loss)
    tf_grad_updates = [lowering.lowered_operation(op) for op in grad_updates]

    # Initializer
    tf_init_vars = \
            FlattenList([lowering.variables[var].laid_out_tensor.all_slices for
                var in graph.trainable_variables])
    tf_init_vars += lowering.variables[pos_enc.operation].laid_out_tensor.all_slices
    init_op = []
    for v in tf_init_vars:
        with tf.device(v.device):
            init_op.append(v.initializer)

    # Training
    cnt = 0
    config = tf.ConfigProto()
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.variable_scope('train'):
        with tf.Session(config=config) as sess:
            dataset.reset_pointer()
            sess.run(init_op)

            tot_time = float(0)
            start = time.time()
            for epoch in range(args['epochs']):
                step = 0

                while True:
                    try:
                        loss_val, *rem = sess.run([tf_loss] + tf_grad_updates)
                        cnt += 1
                        step += 1
                    except tf.errors.OutOfRangeError:
                        break

                    if step % args['display_steps'] == 0 and step > 0:
                        print("Epoch: " + str(epoch) + "; Loss: " + str(loss_val))

                dataset.reset_pointer()
            end = time.time()
            tot_time += (end - start)

    samples_per_sec = (args['batch'] * cnt) / tot_time
    print("Throughout: " + str(samples_per_sec) + " samples / sec")


def main():
    parser = ArgumentParser()

    parser.add_argument('-b', '--batch', type=int, required=False, default=256,
            help="Batch size. (Default: 32)")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors. (Default: 8)")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=3,
            help="No. of epochs")
    parser.add_argument('--display_steps', type=int, required=False, default=10,
            help="No. of epochs")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(3)), 
            help="Strategy to use. 0: DataParallel, \
                    1: Optimized for 1080Ti, \
                    2: Optimized for DGX. \
                    (Default: 0) ")
    parser.add_argument('--src_vocab', type=str, help="Source vocab data file.")
    parser.add_argument('--tgt_vocab', type=str, help="Target vocab data file.")
    parser.add_argument('--src_text', type=str, help="Source text data file.")
    parser.add_argument('--tgt_text', type=str, help="Target text data file.")
    args = vars(parser.parse_args())

    if args['procs'] != 8 and strategy > 0:
        raise NotImplementedError('Current implementation only handles 8 GPUs \
                for model parallel strategies.')

    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(args['procs']))[:-1]
            
    Transformer(args)
    

if __name__ == '__main__':
    main()

