import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf 

import math
from collections import namedtuple
import sys, time, os
import string, random
import argparse

from dataloader import TextDataLoader
from utils import GetMeshImpl
import utils
from mesh_transformations import ReplaceMeshWithIndependentAxes, \
        ReplaceMeshWithDuplicates


def RandName(k=5):
    return ''.join(random.choices(string.ascii_letters + string.ascii_uppercase
        + string.digits, k=k))


def GetShape(dims):
    sh = []
    for d in dims:
        try:
            name, size = d
        except (TypeError, ValueError):
            name, size = RandName(), d
        sh.append(mtf.Dimension(name, size))

    sh = mtf.Shape(sh)
    return sh


class Params():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.nx = 6
        self.max_seq_len = 256
        self.d_model = 1024
        self.heads = 8
        self.d_ff = self.heads * 512
        self.d_k = 128


def CreateMeshes(strategy, src, tgt, params):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}

    def Mesh(idx):
        mesh = mtf.Mesh(graph, 'mesh%d' % idx)
        meshes.append(mesh)
        return mesh

    if strategy == 0: # Data-parallel
        mesh = Mesh(0)
        mesh_to_impl[mesh] = GetMeshImpl([8])

        shape = GetShape([('axis0', params.batch_size), params.max_seq_len])
        mtf_src = mtf.import_tf_tensor(mesh, src, shape)
        mtf_src = mtf.cast(mtf_src, tf.int32)
        mtf_tgt = mtf.import_tf_tensor(mesh, src, shape)
        mtf_tgt = mtf.cast(mtf_tgt, tf.int32)

    elif strategy == 1: # Opt strategy from the tool
        mesh = Mesh(0)
        mesh_to_impl[mesh] = GetMeshImpl([8])
        mesh = Mesh(1)
        mesh_to_impl[mesh] = GetMeshImpl([2, 4])
        mesh = Mesh(2)
        mesh_to_impl[mesh] = GetMeshImpl([4])

        shape = GetShape([params.batch_size, params.max_seq_len])
        mtf_src = mtf.cast(mtf.import_tf_tensor(meshes[0], src, shape), tf.int32)
        mtf_tgt = mtf.cast(mtf.import_tf_tensor(meshes[0], tgt, shape), tf.int32)

    elif strategy == 2: # Strategy from mesh-tensorflow paper
        mesh = Mesh(0)
        mesh_to_impl[mesh] = GetMeshImpl([2, 4])

        shape = GetShape([('axis0', params.batch_size), params.max_seq_len])
        mtf_src = mtf.import_tf_tensor(mesh, src, shape)
        mtf_src = mtf.cast(mtf_src, tf.int32)
        mtf_tgt = mtf.import_tf_tensor(mesh, src, shape)
        mtf_tgt = mtf.cast(mtf_tgt, tf.int32)
        
    else:
        assert False

    return graph, meshes, mesh_to_impl, mtf_src, mtf_tgt


def Transformer(src, tgt, params, src_vocab_size, tgt_vocab_size, args):
    strategy = args.strategy
    graph, meshes, mesh_to_impl, mtf_src, mtf_tgt = CreateMeshes(strategy, src,
            tgt, params)

    # mtf dimensions
    if strategy == 0:
        d_k_dim = mtf.Dimension(RandName(), params.d_k)
        heads_dim = mtf.Dimension(RandName(), params.heads)
        d_ff_dim = mtf.Dimension(RandName(), params.d_ff)
        d_model_dim = mtf.Dimension(RandName(), params.d_model)
        src_vocab_dim = mtf.Dimension(RandName(), src_vocab_size)
        tgt_vocab_dim = mtf.Dimension(RandName(), tgt_vocab_size)
        final_proj_dim = tgt_vocab_dim

    elif strategy == 1:
        src_vocab_size = int(math.ceil(src_vocab_size / 8)) * int(8)
        tgt_vocab_size = int(math.ceil(tgt_vocab_size / 8)) * int(8)
        d_k_dim = mtf.Dimension(RandName(), params.d_k)
        heads_dim = mtf.Dimension('axis0', params.heads)
        d_ff_dim = mtf.Dimension('axis0', params.d_ff)
        d_model_dim = mtf.Dimension('axis0', params.d_model)
        src_vocab_dim = mtf.Dimension(RandName(), src_vocab_size)
        tgt_vocab_dim = mtf.Dimension(RandName(), tgt_vocab_size)
        final_proj_dim = mtf.Dimension('axis0', tgt_vocab_size)

    else:
        # Make the vocabulary size a multiple of mesh size
        src_vocab_size = int(math.ceil(src_vocab_size / 4)) * int(4)
        tgt_vocab_size = int(math.ceil(tgt_vocab_size / 4)) * int(4)
        d_k_dim = mtf.Dimension(RandName(), params.d_k)
        heads_dim = mtf.Dimension('axis1', params.heads)
        d_ff_dim = mtf.Dimension('axis1', params.d_ff)
        d_model_dim = mtf.Dimension(RandName(), params.d_model)
        src_vocab_dim = mtf.Dimension('axis1', src_vocab_size)
        tgt_vocab_dim = mtf.Dimension('axis1', tgt_vocab_size)
        final_proj_dim = tgt_vocab_dim

    seq_len_dim = mtf_src.shape[-1]
    assert mtf_src.shape[-1] == mtf_tgt.shape[-1]

    def ReplaceWithRemoval(x, name=None):
        assert x.mesh == meshes[1]
        new_names = x.shape.dimension_names
        assert new_names[0] == 'axis1'
        new_names[0] = 'axis0'
        return ReplaceMeshWithDuplicates(x, meshes[2], new_names, name=name)

    def ReplaceWithReplication(x, name=None):
        assert x.mesh == meshes[2]
        new_names = x.shape.dimension_names
        assert new_names[0] == 'axis0'
        new_names[0] = 'axis1'
        return  ReplaceMeshWithDuplicates(x, meshes[1], new_names, name=name)

    # Layers
    def EncoderLayer(x, mask, dropout_rate=0.5, name=None):
        assert strategy != 1 \
                or (x.mesh == meshes[2] \
                and x.shape[0].name == 'axis0' \
                and not x.shape[1].name.startswith('axis') \
                and not x.shape[2].name.startswith('axis'))

        with tf.variable_scope(name, default_name='encoder_layer'):
            norm1 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='enc_norm1')

            # Multihead attention
            if strategy == 1:
                norm1 = ReplaceWithReplication(norm1, 'replace_norm1_mesh')
            att = mtf.layers.multihead_attention(norm1, None, mask, d_k_dim,
                    heads_dim, name='enc_multihead_att')
            assert strategy != 1 \
                    or (att.mesh == meshes[1] \
                    and att.shape[0].name == 'axis1' \
                    and not att.shape[1].name.startswith('axis') \
                    and not att.shape[2].name.startswith('axis'))
    
            # Dropout + norm
            if strategy == 1:
                att = ReplaceWithRemoval(att, 'replace_att_mesh')
            assert att.mesh == x.mesh and att.shape == x.shape
            x += mtf.dropout(att, dropout_rate)
            norm2 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='enc_norm2')

            # Feed forward
            if strategy == 1:
                norm2 = ReplaceWithReplication(norm2, 'replace_norm2_mesh')
            ff = mtf.layers.dense_relu_dense(norm2, d_ff_dim, dropout_rate,
                    name='enc_ff')
            if strategy == 1:
                ff = ReplaceWithRemoval(ff, 'replace_ff_mesh')
            assert x.mesh == ff.mesh and x.shape == ff.shape
            x += mtf.dropout(ff, dropout_rate)
    
            assert strategy != 1 \
                    or (x.mesh == meshes[2] \
                    and x.shape[0].name == 'axis0' \
                    and not x.shape[1].name.startswith('axis') \
                    and not x.shape[2].name.startswith('axis'))
            return x

    def DecoderLayer(x, enc_out, enc_mask, dec_mask, dropout_rate=0.5, name=None):
        assert strategy != 1 \
                or (x.mesh == meshes[2] \
                and x.shape[0].name == 'axis0' \
                and not x.shape[1].name.startswith('axis') \
                and not x.shape[2].name.startswith('axis'))
        assert strategy != 1 \
                or (enc_out.mesh == meshes[1] \
                and enc_out.shape[0].name == 'axis1' \
                and not enc_out.shape[1].name.startswith('axis') \
                and not enc_out.shape[2].name.startswith('axis'))

        with tf.variable_scope(name, default_name='decoder_layer'):
            norm1 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm1')

            # Multihead attention 1
            if strategy == 1:
                norm1 = ReplaceWithReplication(norm1, 'replace_norm1_mesh')
            att = mtf.layers.multihead_attention(norm1, None, dec_mask, d_k_dim,
                heads_dim, name='dec_multihead_att1')
            assert strategy != 1 \
                    or (att.mesh == meshes[1] \
                    and att.shape[0].name == 'axis1' \
                    and not att.shape[1].name.startswith('axis') \
                    and not att.shape[2].name.startswith('axis'))

            # Dropout + norm
            if strategy == 1:
                att = ReplaceWithRemoval(att, 'replace_att1_mesh')
            assert att.mesh == x.mesh and att.shape == x.shape
            x += mtf.dropout(att, dropout_rate)
            norm2 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm2')

            # Multihead attention 2
            if strategy == 1:
                norm2 = ReplaceWithReplication(norm2, 'replace_norm2_mesh')
                assert enc_out.mesh == norm2.mesh
            att = mtf.layers.multihead_attention(norm2, enc_out, enc_mask,
                d_k_dim, heads_dim, name='dec_multihead_att2')
            assert strategy != 1 \
                    or (att.mesh == meshes[1] \
                    and att.shape[0].name == 'axis1' \
                    and not att.shape[1].name.startswith('axis') \
                    and not att.shape[2].name.startswith('axis'))

            # Dropout + norm
            if strategy == 1:
                att = ReplaceWithRemoval(att, 'replace_att2_mesh')
            assert att.mesh == x.mesh and att.shape == x.shape
            x += mtf.dropout(att, dropout_rate)
            norm3 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm3')

            # Feed forward
            if strategy == 1:
                norm3 = ReplaceWithReplication(norm3, 'replace_norm3_mesh')
            ff = mtf.layers.dense_relu_dense(norm3, d_ff_dim, dropout_rate,
                    name='dec_ff')
            if strategy == 1:
                ff = ReplaceWithRemoval(ff, 'replace_ff_mesh')
            assert x.mesh == ff.mesh and x.shape == ff.shape
            x += mtf.dropout(ff, dropout_rate)
    
            assert strategy != 1 \
                    or (x.mesh == meshes[2] \
                    and x.shape[0].name == 'axis0' \
                    and not x.shape[1].name.startswith('axis') \
                    and not x.shape[2].name.startswith('axis'))
            return x

    
    # Encoder
    with tf.variable_scope('encoder'):
        # Embedding
        embed = mtf.layers.embedding(mtf_src, src_vocab_dim, d_model_dim,
                tf.float32, name='enc_embedding')
        if strategy == 1:
            embed = ReplaceMeshWithIndependentAxes(embed, meshes[2], ('axis0',
                None, RandName()), name='replace_embed_mesh')

        # Values for positional encoder
        pos = np.array(tuple(range(params.max_seq_len))).reshape(-1, 1)
        val = np.power(10000, ((2 * np.array(tuple(range(params.d_model)))) /
            params.d_model), dtype=float)
        pos_enc_values = np.divide(pos, val, dtype=np.float32)
        np.sin(pos_enc_values[:,::2], out=pos_enc_values[:,::2],
                dtype=np.float32)
        np.cos(pos_enc_values[:,1::2], out=pos_enc_values[:,1::2],
                dtype=np.float32)
        
        # positional encoder
        pos_enc = mtf.get_variable(embed.mesh, 'pos_enc',
                shape=mtf.Shape([seq_len_dim, embed.shape[-1]]),
                dtype=tf.float32,
                initializer=tf.constant_initializer(pos_enc_values),
                trainable=False)
        assert embed.shape[1:] == pos_enc.shape.dims
        x = (embed * math.sqrt(params.d_model)) + pos_enc

        # Encoder layers
        enc_mask = None
        for i in range(params.nx):
            x = EncoderLayer(x, enc_mask, name='enc_layer_%d' %i)
        enc_output = mtf.layers.layer_norm(x, dim=x.shape[-1],
                name='enc_final_norm')

    # Decoder
    with tf.variable_scope('decoder'):
        # Embedding + positional encoder
        embed = mtf.layers.embedding(mtf_tgt, tgt_vocab_dim, d_model_dim,
                tf.float32, name='dec_embedding')
        if strategy == 1:
            assert not enc_output.shape[-1].name.startswith('axis')
            embed = ReplaceMeshWithIndependentAxes(embed, meshes[2], ('axis0',
                None, enc_output.shape[-1].name), name='replace_embed_mesh')
        assert embed.shape[1:] == pos_enc.shape.dims
        x = (embed * math.sqrt(params.d_model)) + pos_enc

        # Decoder layers
        enc_output = mtf.rename_dimension(enc_output, enc_output.shape[1].name,
                RandName())
        if strategy == 1:
            enc_output = ReplaceWithReplication(enc_output,
                    'replace_enc_out_mesh')
        dec_mask = None
        for i in range(params.nx):
            x = DecoderLayer(x, enc_output, enc_mask, dec_mask,
                    name='dec_layer_%d' % i)
        dec_output = mtf.layers.layer_norm(x, dim=x.shape[-1],
                name='dec_final_norm')

    # Loss function
    with tf.variable_scope('loss'):
        if strategy == 1:
            assert dec_output.mesh == meshes[2]
            assert dec_output.shape[0].name == 'axis0'
            dec_output = ReplaceMeshWithIndependentAxes(dec_output, meshes[0],
                    (mtf_tgt.shape[0].name, None, None),
                    name='replace_dec_out_mesh')
        out = mtf.layers.dense(dec_output, final_proj_dim, use_bias=False,
                name='final_projection')
        one_hot_labels = mtf.one_hot(mtf_tgt, out.shape[-1], dtype=out.dtype)
        out = mtf.layers.softmax_cross_entropy_with_logits(out, one_hot_labels,
                out.shape[-1])
        loss = mtf.reduce_mean(out)

    with tf.variable_scope('optimize'):
        grads = mtf.gradients([loss], [v.outputs[0] for v in
            graph.trainable_variables])
        lr = 0.01
        opt = mtf.optimize.SgdOptimizer(lr)
        grad_updates = opt.apply_grads(grads, graph.trainable_variables)

    print('Lowering mtf ops...', flush=True)
    lowering = mtf.Lowering(graph, mesh_to_impl)
    print('Finished lowering.', flush=True)
    tf_loss = lowering.export_to_tf_tensor(loss)
    tf_grad_updates = [lowering.lowered_operation(op) for op in grad_updates]

    # Initializer
    tf_init_vars = \
            utils.FlattenList([lowering.variables[var].laid_out_tensor.all_slices
                for var in graph.all_variables])
    init_ops = []
    for v in tf_init_vars:
        with tf.device(v.device):
            init_ops.append(v.initializer)

    return init_ops, tf_loss, tf_grad_updates


def main():
    parser = argparse.ArgumentParser(formatter_class =
            argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b', '--batch', type=int, required=False, default=64,
            help="Batch size")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=3,
            help="No. of epochs")
    parser.add_argument('--max_steps', type=int, required=False, default=500,
            help='Maximum no. of steps to execute')
    parser.add_argument('--display_steps', type=int, required=False, default=10,
            help="No. of epochs")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(3)), 
            help="Strategy to use. 0: DataParallel, \
                    1: Optimized, \
                    2: Expert defined.")
    parser.add_argument('--src_vocab', type=str, help="Source vocab data file.")
    parser.add_argument('--tgt_vocab', type=str, help="Target vocab data file.")
    parser.add_argument('--src_text', type=str, help="Source text data file.")
    parser.add_argument('--tgt_text', type=str, help="Target text data file.")
    args = parser.parse_args()
    params = Params(args.batch)

    if args.procs != 8:
        raise NotImplementedError('Current implementation only handles 8 GPUs.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(args.procs))[:-1]

    for arg, val in vars(args).items():
        print(str(arg) + ": " + str(val))
    print()
            
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
    print("Source vocab size: %d" % src_vocab_size)
    print("Target vocab size: %d" % tgt_vocab_size)

    init_ops, loss_op, grad_ops = Transformer(enc_inputs, dec_inputs, params,
            src_vocab_size, tgt_vocab_size, args)
 
    cnt = 0
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
    config = tf.ConfigProto()
    #config = tf.ConfigProto(log_device_placement=True,
    #        allow_soft_placement=False)
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.variable_scope('train'):
        with tf.Session(config=config) as sess:
            dataset.reset_pointer()
            sess.run(init_ops)

            tot_time = float(0)
            start = time.time()
            for epoch in range(args.epochs):
                step = 0

                while True:
                    try:
                        loss_val, *_ = sess.run([loss_op] + grad_ops,
                                options=run_options)
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

