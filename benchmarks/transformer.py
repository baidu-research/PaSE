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
    def __init__(self, x, axis, mesh, new_shape=None, name=None):
        assert len(axis) == 2
        assert axis[0] is not None
        assert all(a is None or a < x.shape.ndims for a in axis)
        self.old_mesh = x.mesh
        self.new_axis = axis
        self.old_shape = x.shape
        super().__init__([x], mesh=mesh, name=name or 'replace_mesh1_with_mesh2')

        # Find the current axis along which the tensor is sliced
        self.old_axis = None
        for i, a in enumerate(x.shape.dims):
            if a.name == 'axis0':
                self.old_axis = i
                break
        assert self.old_axis is None or all(self.old_axis != axis[i] for i
                in range(2))

        # Output tensor
        if new_shape is None:
            new_shape = x.shape
            if self.old_axis is not None:
                new_shape = utils.RenameDim(x.shape, self.old_axis, RandName())
            for i, a in enumerate(axis):
                if a is not None:
                    new_shape = utils.RenameDim(new_shape, a, 'axis%d' % i)
        else:
            assert new_shape[axis[0]].name == 'axis0'
            assert axis[1] is None or new_shape[axis[1]].name -- 'axis1'
            assert self.old_axis is None or \
                    not new_shape[self.old_axis].name.startswith('axis')
        self._outputs = [mtf.Tensor(self, new_shape, x.dtype)]

    def gradient(self, grad_ys):
        return Mesh1ToMesh0Operation(grad_ys[0], self.old_axis,
                self.old_mesh, self.old_shape).outputs

    def lower(self, lowering):
        input_slices = \
                lowering.tensors[self.inputs[0]].to_laid_out_tensor().tensor_list
        if self.old_axis is None:
            input_slices = input_slices[:1]

        # Split along new axis
        split_slices = []
        for i, s in enumerate(input_slices):
            with tf.device(s.device):
                slices = tf.split(s, 4, axis=self.new_axis[0],
                        name='split_along_mesh0_%d' % i)
                tmp_slices = []
                for j, t in enumerate(slices):
                    if self.new_axis[1] is not None:
                        tmp_slices += tf.split(t, 2, axis=self.new_axis[1],
                                name='split_along_mesh0_%d_%d' % {i, j})
                    else:
                        tmp_slices += [t, t]
                split_slices.append(tmp_slices)

        # Concatenate along old axis
        if self.old_axis is not None:
            out_slices = []
            for i, ta in enumerate(zip(*split_slices)):
                with tf.device('/device:GPU:%d' % i):
                    out_slices.append(tf.concat(ta, axis=self.old_axis,
                        name='concat_%d' % i))
        else:
            old_slices = split_slices

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(out_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def Mesh0ToMesh1(x, axis, new_mesh, name=None):
    return Mesh0ToMesh1Operation(x, axis, new_mesh, name=name).outputs[0]


class Mesh1ToMesh0Operation(mtf.Operation):
    def __init__(self, x, axis, mesh, new_shape=None, name=None):
        assert axis is None or axis < x.shape.ndims
        self.old_mesh = x.mesh
        self.new_axis = axis
        self.old_shape = x.shape
        super().__init__([x], mesh=mesh, name=name or 'replace_mesh2_with_mesh1')

        # Find the axes along which 'x' is split
        self.old_axis = [None, None]
        for i, a in enumerate(x.shape.dims):
            if a.name == 'axis0':
                self.old_axis[0] = i
            elif a.name == 'axis1':
                self.old_axis[1] = i
        assert self.old_axis[0] is not None
        assert axis != self.old_axis[0] and (self.old_axis[1] is None or axis !=
                self.old_axis[1])

        # Output tensor
        if new_shape is None:
            new_shape = x.shape
            if axis is not None:
                new_shape = utils.RenameDim(new_shape, axis, 'axis0')
            for a in self.old_axis:
                if a is not None:
                    new_shape = utils.RenameDim(new_shape, a, RandName())
        else:
            assert axis is None or new_shape[axis].name == 'axis0'
            assert not new_shape[self.old_axis[0]].name.startswith('axis')
            assert self.old_axis[1] is None or \
                    not new_shape[self.old_axis[1]].name.startswith('axis')
        self._outputs = [mtf.Tensor(self, new_shape, x.dtype)]

    def gradient(self, grad_ys):
        return Mesh0ToMesh1Operation(grad_ys[0], self.old_axis,
                self.old_mesh, self.old_shape).outputs

    def lower(self, lowering):
        input_slices = \
                lowering.tensors[self.inputs[0]].to_laid_out_tensor().tensor_list
        if self.old_axis[1] is None:
            input_slices = input_slices[::2]

        # Split along new axis
        split_slices = []
        if self.new_axis is not None:
            for i, s in enumerate(input_slices):
                with tf.device(s.device):
                    split_slices.append(tf.split(s, 8, axis=self.new_axis,
                        name='split_along_mesh1_%d' % i))
        else:
            split_slices.append([s] * 8 for s in input_slices)

        # Concatenate along old axis
        if self.old_axis[1] is not None:
            assert len(split_slices) == 8
            tmp_slices = []
            for i, s1, s2 in enumerate(zip(
                split_slices[0::2], split_slices[1::2])):
                ta = []
                for j, t in enumerate(zip(s1, s2)):
                    with tf.device('/device:GPU:%d' % j):
                        ta.append(tf.concat(t, self.old_axis[1],
                            name='concat_x_%d_%d' % {i,j}))
                tmp_slices.append(ta)
            split_slices = tmp_slices
        assert len(split_slices) == 4
        out_slices = []
        for i, t in enumerate(zip(*split_slices)):
            with tf.device('/device:GPU:%d' % i):
                out_slices.append(tf.concat(t, self.old_axis[0],
                    name='concat_%d' % i))

        # Lower the new slices
        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(out_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def Mesh1ToMesh0(x, axis, new_mesh, name=None):
    return Mesh1ToMesh0Operation(x, axis, new_mesh, name=name).outputs[0]


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
        heads_dim = mtf.Dimension('axis1', params.heads)
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

    # Layers
    def DenseReLUDense(x,
                         hidden_channels,
                         dropout=0.0,
                         dropout_broadcast_dims=None,
                         master_dtype=tf.float32,
                         slice_dtype=tf.float32, name=None):
        assert strategy != 1 \
                or (x.mesh == meshes[0] \
                and not x.shape[0].name.startswith('axis') \
                and not x.shape[1].name.startswith('axis') \
                and not x.shape[2].name.startswith('axis') \
                and hidden_channels.name == 'axis0')

        io_channels = x.shape.dims[-1]
        with tf.variable_scope(name, default_name="dense_relu_dense"):
            h = mtf.layers.dense(x, hidden_channels,
                      use_bias=False, activation=mtf.relu,
                      master_dtype=master_dtype, slice_dtype=slice_dtype, name="wi")
            if dropout != 0.0:
                h = mtf.dropout(h, 1.0 - dropout, noise_shape=h.shape -
                        dropout_broadcast_dims)

            if strategy == 1:
                with tf.variable_scope('rename_h'):
                    h = mtf.rename_dimension(h, h.shape[-1].name, RandName())
                io_channels = io_channels._replace(name='axis0')

            return mtf.layers.dense(h, io_channels, use_bias=False, activation=None,
                       master_dtype=master_dtype, slice_dtype=slice_dtype,
                       name="wo")
    
    def EncoderLayer(x, mask, dropout_rate=0.5, name=None):
        assert strategy != 1 \
                or (x.mesh == meshes[0] \
                and not x.shape[0].name.startswith('axis') \
                and not x.shape[1].name.startswith('axis') \
                and x.shape[2].name == 'axis0')

        with tf.variable_scope(name, default_name='encoder_layer'):
            norm1 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='enc_norm1')
            if strategy == 1:
                norm1 = Mesh0ToMesh1(norm1, (0, None), meshes[1])

            # Multihead attention
            att = mtf.layers.multihead_attention(norm1, None, mask, d_k_dim,
                    heads_dim, name='enc_multihead_att')
            assert strategy != 1 \
                    or (att.mesh == meshes[1] \
                    and att.shape[0].name == 'axis0' \
                    and not att.shape[1].name.startswith('axis') \
                    and not att.shape[2].name.startswith('axis'))
    
            # Dropout + norm
            if strategy == 1:
                att = Mesh1ToMesh0(att, 2, meshes[0])
                att = mtf.rename_dimension(att, att.shape[0].name,
                        x.shape[0].name)
            assert att.shape == x.shape
            x += mtf.dropout(att, dropout_rate)
            norm2 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='enc_norm2')
            if strategy == 1:
                assert norm2.shape[-1].name == 'axis0'
                norm2 = mtf.rename_dimension(norm2, norm2.shape[-1].name,
                        RandName())

            # Feed forward
            ff = DenseReLUDense(norm2, d_ff_dim, dropout_rate, name='enc_ff')
            assert x.shape == ff.shape
            x += mtf.dropout(ff, dropout_rate)
    
            assert strategy != 1 \
                    or (x.mesh == meshes[0] \
                    and not x.shape[0].name.startswith('axis') \
                    and not x.shape[1].name.startswith('axis') \
                    and x.shape[2].name == 'axis0')
            return x

    def DecoderLayer(x, enc_out, enc_mask, dec_mask, dropout_rate=0.5, name=None):
        assert strategy != 1 \
                or (x.mesh == meshes[0] \
                and not x.shape[0].name.startswith('axis') \
                and not x.shape[1].name.startswith('axis') \
                and x.shape[2].name == 'axis0')
        assert strategy != 1 \
                or (enc_out.mesh == meshes[1] \
                and enc_out.shape[0].name == 'axis0' \
                and not enc_out.shape[1].name.startswith('axis') \
                and not enc_out.shape[2].name.startswith('axis'))

        with tf.variable_scope(name, default_name='decoder_layer'):
            if strategy == 1:
                x = Mesh0ToMesh1(x, (0, None), meshes[1])
                x = mtf.rename_dimension(x, x.shape[-1].name,
                        enc_out.shape[-1].name)
            norm1 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm1')

            enc_out = mtf.rename_dimension(enc_out, enc_out.shape[1].name,
                    RandName())

            # Multihead attention 1
            att = mtf.layers.multihead_attention(norm1, None, dec_mask, d_k_dim,
                heads_dim, name='dec_multihead_att1')
            assert strategy != 1 \
                    or (att.mesh == meshes[1] \
                    and att.shape[0].name == 'axis0' \
                    and not att.shape[1].name.startswith('axis') \
                    and not att.shape[2].name.startswith('axis'))

            # Dropout + norm
            assert att.shape == x.shape
            x += mtf.dropout(att, dropout_rate)
            norm2 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm2')

            # Multihead attention 2
            att = mtf.layers.multihead_attention(norm2, enc_out, enc_mask,
                d_k_dim, heads_dim, name='dec_multihead_att2')
            assert strategy != 1 \
                    or (att.mesh == meshes[1] \
                    and att.shape[0].name == 'axis0' \
                    and not att.shape[1].name.startswith('axis') \
                    and not att.shape[2].name.startswith('axis'))

            # Dropout + norm
            assert att.shape == x.shape
            x += mtf.dropout(att, dropout_rate)
            if strategy == 1:
                x = Mesh1ToMesh0(x, 2, meshes[0])
            norm3 = mtf.layers.layer_norm(x, dim=x.shape[-1], name='dec_norm3')
            if strategy == 1:
                norm3 = mtf.rename_dimension(norm3, norm3.shape[-1].name,
                        RandName())

            ff = DenseReLUDense(norm3, d_ff_dim, dropout_rate, name='dec_ff')
            assert x.shape == ff.shape
            x += mtf.dropout(ff, dropout_rate)
    
            assert strategy != 1 \
                    or (x.mesh == meshes[0] \
                    and not x.shape[0].name.startswith('axis') \
                    and not x.shape[1].name.startswith('axis') \
                    and x.shape[2].name == 'axis0')
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
            x = Mesh0ToMesh1(x, (0, None), meshes[1], name='rename_encoder_output')
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
            dec_output = mtf.rename_dimension(dec_output,
                    dec_output.shape[0].name, mtf_tgt.shape[0].name)
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

