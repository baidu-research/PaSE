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
        mesh_to_impl[mesh] = GetMeshImpl([4, 2])

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


class Mesh0ToMesh1Operation(mtf.Operation):
    def __init__(self, xs, shape, axis, mesh, name=None):
        assert len(axis) == 2
        self.old_mesh = xs[0].mesh
        self.old_shape = shape

        self.concat_axis = None
        for i, a in enumerate(shape.dims):
            if a.name == 'axis0':
                self.concat_axis = i
                break
        assert self.concat_axis is not None
        assert all(self.concat_axis != axis[i] for i in range(2))

        super().__init__(xs, mesh=mesh, name=name or 'replace_mesh1_with_mesh2')
        new_shape = utils.RenameDim(shape, self.concat_axis, RandName())
        for i, a in enumerate(axis):
            if a is not None:
                new_shape = utils.RenameDim(new_shape, a, 'axis%d' % i)
        self._outputs = [mtf.Tensor(self, new_shape, xs[0].dtype)]

    def gradient(self, grad_ys):
        return Mesh1ToMesh0Operation(grad_ys, self.old_shape, self.concat_axis,
                self.old_mesh).outputs

    def lower(self, lowering):
        slices = []
        for i, s in enumerate(self.inputs):
            with tf.device('/device:GPU:%d' % i):
                tensor_list = \
                        lowering.tensors[s].to_laid_out_tensor().tensor_list
                slices.append(tf.concat(tensor_list, axis=self.concat_axis))

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def Mesh0ToMesh1(x, new_mesh, axis, name=None):
    assert len(axis) == 2
    assert all(a < x.shape.ndims for a in axis)

    split_x = mtf.split(x, x.shape[axis[0]], 4, name='split_along_mesh2_x')
    if axis[1] is not None:
        slices = []
        for i, s in enumerate(split_x):
            slices += mtf.split(s, s.shape[axis[1]], 2,
                    name='split_along_mesh2_y_%d' % i)
    else:
        slices = split_x

    return Mesh0ToMesh1Operation(slices, x.shape, axis, new_mesh,
            name).outputs[0]


class Mesh1ToMesh0Operation(mtf.Operation):
    def __init__(self, xs, shape, axis, mesh, name=None):
        assert len(xs) == 8
        self.old_mesh = xs[0].mesh
        self.old_shape = shape

        self.axis = (None, None)
        for i, a in enumerate(shape.dims):
            if a.name == 'axis0':
                self.axis[0] = i
            elif a.name == 'axis1':
                self.axis[1] = i
        assert self.axis[0] is not None
        assert axis != self.axis[0] and (self.axis[1] is None or axis !=
                self.axis[1])

        super().__init__(xs, mesh=mesh, name=name or 'replace_mesh2_with_mesh1')
        new_shape = utils.RenameDim(shape, axis, 'axis0')
        for a in self.axis:
            if a is not None:
                new_shape = utils.RenameDim(new_shape, a, RandName())
        self._outputs = [mtf.Tensor(self, new_shape, xs[0].dtype)]

    def gradient(self, grad_ys):
        assert len(grad_ys) == 8
        return Mesh0ToMesh1Operation(grad_ys, self.old_shape, self.axis,
                self.old_mesh).outputs

    def lower(self, lowering):
        slices = []
        for i, s in enumerate(self.inputs):
            with tf.device('/device:GPU:%d' % i):
                tensor_list = \
                        lowering.tensors[s].to_laid_out_tensor().tensor_list
                slices_x = []
                assert len(tensor_list) == 8
                if self.axis[1] is not None:
                    for t in zip(tensor_list[0::2], tensor_list[1::2]):
                        slices_x.append(tf.concat(t, self.axis[1]))
                else:
                    slices_x = tensor_list
                slices.append(tf.concat(slices_x, self.axis[0]))

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def Mesh1ToMesh0(x, new_mesh, axis, name=None):
    assert axis < x.shape.ndims
    slices = mtf.split(x, x.shape[axis], 8, name='split_along_mesh1_x')
    return Mesh1ToMesh0Operation(slices, x.shape, axis, new_mesh,
            name).outputs[0]


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

    elif strategy == 1:
        d_k_dim = mtf.Dimension(RandName(), params.d_k)
        heads_dim = mtf.Dimension('axis1', params.heads)
        d_ff_dim = mtf.Dimension('axis0', params.d_ff)
        d_model_dim = mtf.Dimension('axis0', params.d_model)
        src_vocab_dim = mtf.Dimension(RandName(), src_vocab_size)
        tgt_vocab_dim = mtf.Dimension('axis0', tgt_vocab_size)

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

    seq_len_dim = mtf_src.shape[-1]
    assert mtf_src.shape[-1] == mtf_tgt.shape[-1]

    # Layers
    '''
    def MultiHeadAttn(query_antecedent,
                            memory_antecedent,
                            mask,
                            kv_channels,
                            heads,
                            dropout=0.0,
                            dropout_broadcast_dims=None,
                            master_dtype=tf.float32,
                            slice_dtype=tf.float32,
                            name="multihead_attention"):
        batch_dims = query_antecedent.shape.dims[:-2]
        query_length, io_channels = query_antecedent.shape.dims[-2:]
        with tf.variable_scope(name, default_name="multihead_attention",
                values=[query_antecedent, memory_antecedent]):
            wq, wk, wv, wo = multihead_attention_vars(
                    query_antecedent.mesh, heads, io_channels, kv_channels,
                    master_dtype, slice_dtype, query_antecedent.dtype)
            if memory_antecedent is None:
                memory_antecedent = mtf.rename_length_to_memory_length(
                        query_antecedent, query_length.name)
            memory_batch_dims = memory_antecedent.shape.dims[:-2]
            memory_length, memory_channels = memory_antecedent.shape.dims[-2:]
            if memory_batch_dims != batch_dims:
                raise ValueError("memory batch must equal query batch")
            if memory_channels != io_channels:
                raise ValueError("memory channels must equal query channels")
            q = mtf.einsum(
                [query_antecedent, wq],
                mtf.Shape(batch_dims + [heads, query_length, kv_channels]))
            k = mtf.einsum(
                [memory_antecedent, wk],
                mtf.Shape(batch_dims + [heads, memory_length, kv_channels]))
            v = mtf.einsum(
                [memory_antecedent, wv],
                mtf.Shape(batch_dims + [heads, memory_length, kv_channels]))

            o = mtf.layers.dot_product_attention(
                q, k, v, mask, dropout, dropout_broadcast_dims)

            if strategy == 1:
                assert o.shape[1].name == 'axis1'
                o = mtf.rename_dimension(o, o.shape[1].name, RandName())
                wo = mtf.rename_dimension(wo, wo.shape[1].name, 'axis1')

            return mtf.einsum(
                [o, wo], mtf.Shape(batch_dims + [query_length, io_channels]))
    '''

    def DenseReLUDense(x,
                         hidden_channels,
                         dropout=0.0,
                         dropout_broadcast_dims=None,
                         master_dtype=tf.float32,
                         slice_dtype=tf.float32, name=None):
        with tf.variable_scope(name, default_name="dense_relu_dense"):
            io_channels = x.shape.dims[-1]
            h = mtf.layers.dense(x, hidden_channels,
                      use_bias=False, activation=mtf.relu,
                      master_dtype=master_dtype, slice_dtype=slice_dtype, name="wi")
            if dropout != 0.0:
                h = mtf.dropout(h, 1.0 - dropout,
                        noise_shape=h.shape - dropout_broadcast_dims)

            if strategy == 1:
                assert h.shape[-1].name == 'axis0'
                with tf.variable_scope('rename_h'):
                    h = mtf.rename_dimension(h, h.shape[-1].name, RandName())
                io_channels = io_channels._replace(name='axis0')

            return mtf.layers.dense(h, io_channels, use_bias=False, activation=None,
                       master_dtype=master_dtype, slice_dtype=slice_dtype,
                       name="wo")
    
    def EncoderLayer(x, mask, dropout_rate=0.5, name=None):
        with tf.variable_scope(name, default_name='encoder_layer'):
            norm1 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='enc_norm1')
            if strategy == 1:
                norm1 = Mesh0ToMesh1(norm1, meshes[1], (0, None))

            att = mtf.layers.multihead_attention(norm1, None, mask, d_k_dim,
                    heads_dim, name='enc_multihead_att')
            x += mtf.dropout(att, dropout_rate)
    
            if strategy == 1:
                x = Mesh1ToMesh0(x, meshes[0], 3)
            norm2 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='enc_norm2')
            ff = DenseReLUDense(norm2, d_ff_dim, dropout_rate, name='enc_ff')
            x += mtf.dropout(ff, dropout_rate)
    
            return x

    def DecoderLayer(x, enc_out, enc_mask, dec_mask, dropout_rate=0.5, name=None):
        with tf.variable_scope(name, default_name='decoder_layer'):
            norm1 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm1')
            if strategy == 1:
                embed = Mesh0ToMesh1(embed, meshes[1], (0, None))

            att = mtf.layers.multihead_attention(norm1, None, dec_mask, d_k_dim,
                heads_dim, name='dec_multihead_att1')
            x += mtf.dropout(att, dropout_rate)
    
            norm2 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm2')
            if strategy == 1:
                norm2 = mtf.rename_dimension(norm2, norm2.shape[1].name, 'axis1')
            enc_out = mtf.rename_dimension(enc_out, enc_out.shape[1].name,
                    RandName())
            att = mtf.layers.multihead_attention(norm2, enc_out, enc_mask,
                d_k_dim, heads_dim, name='dec_multihead_att2')
            x += mtf.dropout(att, dropout_rate)
    
            if strategy == 1:
                x = Mesh1ToMesh0(x, meshes[0], 3)
            norm3 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm3')
            ff = DenseReLUDense(norm3, d_ff_dim, dropout_rate, name='dec_ff')
            x += mtf.dropout(ff, dropout_rate)
    
            return x

    
    # Encoder
    with tf.variable_scope('encoder'):
        # Embedding
        embed = mtf.layers.embedding(mtf_src, src_vocab_dim, d_model_dim,
                tf.float32, name='enc_embedding')

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
                shape=mtf.Shape([seq_len_dim, d_model_dim]), dtype=tf.float32,
                initializer=tf.constant_initializer(pos_enc_values),
                trainable=False)
        x = (embed * math.sqrt(params.d_model)) + pos_enc

        # Encoder layers
        enc_mask = None
        for i in range(params.nx):
            x = EncoderLayer(x, enc_mask, name='enc_layer_%d' %i)
        if strategy == 1:
            x = Mesh0ToMesh1(x, meshes[1], (0, 1), name='rename_encoder_output')
        enc_output = mtf.layers.layer_norm(x, dim=x.shape[-1],
                name='enc_final_norm')

    # Decoder
    with tf.variable_scope('decoder'):
        # Embedding + positional encoder
        embed = mtf.layers.embedding(mtf_tgt, tgt_vocab_dim, d_model_dim,
                tf.float32, name='dec_embedding')
        x = (embed * math.sqrt(params.d_model)) + pos_enc

        # Decoder layers
        dec_mask = None
        for i in range(params.nx):
            x = DecoderLayer(x, enc_output, enc_mask, dec_mask,
                    name='dec_layer_%d' % i)
        dec_output = mtf.layers.layer_norm(x, dim=x.shape[-1],
                name='dec_final_norm')

    # Loss function
    with tf.variable_scope('loss'):
        if strategy == 1:
            assert dec_output.shape[-1].name == 'axis0'
            dec_output = mtf.rename_dimension(dec_output, 'axis0', RandName())
        out = mtf.layers.dense(dec_output, tgt_vocab_dim, use_bias=False,
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
                    except tf.errors.OutOfRangeError:
                        break

                    if step % args.display_steps == 0 and step > 0:
                        print("Epoch: " + str(epoch) + "; Loss: " + str(loss_val))

                dataset.reset_pointer()
            end = time.time()
            tot_time += (end - start)

    samples_per_sec = (args.batch * cnt) / tot_time
    print("Throughout: " + str(samples_per_sec) + " samples / sec")


if __name__ == '__main__':
    main()

