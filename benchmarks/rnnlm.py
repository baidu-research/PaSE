import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf 

import math
import datetime
import sys, time, os
import string, random
import argparse

from dataloader import TextDataLoader
from utils import GetMeshImpl
import utils
from mesh_transformations import ReplaceMeshWithIndependentAxes, \
        ReplaceMeshWithDuplicates, ReplaceMesh


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
        self.num_units = 2048
        self.max_seq_len = 64
        self.num_layers = 2


class ConcatOperation(mtf.ConcatOperation):
    def gradient(self, grad_ys):
        dy = grad_ys[0]
        return split(dy, self.outputs[0].shape.dims[self._axis],
                self._input_sizes)

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        def slicewise_fn(*args):
            return tf.concat(args, axis=self._axis, name="concat")
        y = mesh_impl.slicewise(
                slicewise_fn, *[lowering.tensors[x] for x in self._inputs])
        lowering.set_tensor_lowering(self.outputs[0], y)


class SplitOperation(mtf.SplitOperation):
    def gradient(self, grad_ys):
        return [concat(grad_ys, self._split_dim.name)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        ma = mesh_impl.tensor_dimension_to_mesh_axis(self._split_dim)
        if ma is not None:
            axis_size = mesh_impl.shape[ma].size
            output_sizes = [s // axis_size for s in self._output_sizes]
        else:
            output_sizes = self._output_sizes
        def slicewise_fn(x):
            return tuple(tf.split(x, output_sizes, axis=self._axis))
        values = mesh_impl.slicewise(
                slicewise_fn, lowering.tensors[self.inputs[0]])
        for t, v in zip(self._outputs, values):
            lowering.set_tensor_lowering(t, v)


def concat(xs, concat_dim_name, name=None):
    return ConcatOperation(xs, concat_dim_name, name).outputs[0] 


def split(x, split_dim, num_or_size_splits, name=None):
    return SplitOperation(x, split_dim, num_or_size_splits, name=name).outputs


def CreateMeshes(strategy, src, tgt, params):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}

    def Mesh():
        mesh = mtf.Mesh(graph, 'mesh%d' % Mesh.idx)
        meshes.append(mesh)
        Mesh.idx += 1
        return mesh
    Mesh.idx = 0

    if strategy == 0: # Data-parallel
        mesh = Mesh()
        mesh_to_impl[mesh] = GetMeshImpl([8])

        shape = GetShape([('axis0', params.batch_size), params.max_seq_len])
        mtf_src = mtf.cast(mtf.import_tf_tensor(mesh, src, shape), tf.int32)
        mtf_tgt = mtf.cast(mtf.import_tf_tensor(mesh, src, shape), tf.int32)

    elif strategy == 1:
        mesh_to_impl[Mesh()] = GetMeshImpl([4, 2])
        mesh_to_impl[Mesh()] = GetMeshImpl([2, 2])
        mesh_to_impl[Mesh()] = GetMeshImpl([8])
        mesh_to_impl[Mesh()] = GetMeshImpl([2], [0, 2])

        shape = GetShape([params.batch_size, params.max_seq_len]) 
        mtf_src = mtf.cast(mtf.import_tf_tensor(meshes[0], src, shape), tf.int32)
        mtf_tgt = mtf.cast(mtf.import_tf_tensor(meshes[2], tgt, shape), tf.int32)

    elif strategy == 2:
        mesh_to_impl[Mesh()] = GetMeshImpl([4], [0, 1, 2, 3])
        mesh_to_impl[Mesh()] = GetMeshImpl([4], [4, 5, 6, 7])

        shape = GetShape([('axis0', params.batch_size), params.max_seq_len]) 
        mtf_src = mtf.cast(mtf.import_tf_tensor(meshes[0], src, shape), tf.int32)
        mtf_tgt = mtf.cast(mtf.import_tf_tensor(meshes[0], tgt, shape), tf.int32)

    else:
        assert False

    return graph, meshes, mesh_to_impl, mtf_src, mtf_tgt


def RNNLM(src, tgt, params, vocab_size, args):
    strategy = args.strategy
    graph, meshes, mesh_to_impl, mtf_src, mtf_tgt = CreateMeshes(strategy, src,
            tgt, params)

    if strategy == 0 or strategy == 2:
        vocab_dim = mtf.Dimension(RandName(), vocab_size)
        embed_dim = mtf.Dimension(RandName(), params.num_units)
        rnn_out_dims = mtf.Dimension(RandName(), params.num_units)
        final_proj_dim = vocab_dim

    elif strategy == 1:
        vocab_dim = mtf.Dimension('axis1', vocab_size)
        embed_dim = mtf.Dimension('axis0', params.num_units)
        rnn_out_dims = [mtf.Dimension('axis1', params.num_units),
                mtf.Dimension('axis0', params.num_units)]
        final_proj_dim = mtf.Dimension('axis0', vocab_size)

    else:
        assert False

    assert mtf_src.shape[-1] == mtf_tgt.shape[-1]

    class LSTMCell():
        def __init__(self, mesh, shape, mesh_repl_fn=None, output_dim=None,
                h=None, name=None):
            assert shape.to_integer_list == [params.batch_size, params.num_units]
            assert output_dim.size == params.num_units
            self.name = name or 'lstm'
            self.mesh = mesh
            self.shape = shape
            self.mesh_repl_fn = mesh_repl_fn or (lambda x: x)

            # Dimensions for weight matrix
            k_dim = mtf.Dimension(shape[-1].name, 2 * params.num_units)
            n_dim = mtf.Dimension(output_dim.name, 4 * params.num_units)

            # Initial h, c, and w
            self.h = mtf.zeros(mesh, shape) if h is None else h
            self.c = mtf.zeros(mesh, mtf.Shape([shape[0], output_dim]))
            self.w = mtf.get_variable(mesh, f'{self.name}_w', mtf.Shape([k_dim,
                n_dim]))

        def __call__(self, x):
            x = self.mesh_repl_fn(x)
            assert x.mesh == self.mesh and x.shape == self.shape
            assert self.h.mesh == self.mesh
            if self.h.shape != self.shape:
                self.h = mtf.reshape(self.h, self.shape)

            # Concatenate x, and h
            xh = concat((x, self.h), x.shape[-1].name,
                    name=f'{self.name}_concat')
 
            # GEMM
            ifgo = mtf.einsum([xh, self.w], reduced_dims=[xh.shape[-1]],
                    name=f'{self.name}_ifgo')
            i, f, g, o = split(ifgo, ifgo.shape[-1], 4,
                    name=f'{self.name}_split_ifgo')

            # Activations
            i, f, o = (mtf.sigmoid(t) for t in (i, f, o))
            g = mtf.tanh(g)

            # Elementwise ops
            assert self.c.shape == f.shape
            self.c = (f * self.c) + (i * g)
            self.h = o * mtf.tanh(self.c)

            return self.h

    class RNNStack():
        def __init__(self, meshes, shapes, repl_fns, output_dims, name=None):
            Replicate = lambda x: x if isinstance(x, list) \
                    else [x] * params.num_layers
            meshes, shapes, repl_fns, output_dims = (Replicate(x)
                    for x in [meshes, shapes, repl_fns, output_dims])
            self.shapes = shapes

            assert len(meshes) == len(shapes) == len(repl_fns) \
                    == len(output_dims)  == params.num_layers

            self.name = name or 'rnn_stack'
            self.layers = [LSTMCell(mesh, shape, repl_fn, dim,
                name=f'{self.name}_lstm_{i}'
                ) for i, (mesh, shape, repl_fn, dim) in enumerate(
                        zip(meshes, shapes, repl_fns, output_dims))]

        def __call__(self, xs):
            assert len(xs) == params.max_seq_len

            ys = []
            for x in xs:
                # First layer
                with tf.variable_scope(f'{self.name}_layer_0'):
                    x = self.layers[0](x)

                # Second layers
                with tf.variable_scope(f'{self.name}_layer_1'):
                    x = self.layers[1](x)
                ys.append(x)
            return ys

        @property
        def hs(self):
            return [lstm.h for lstm in self.layers]

    def RNN(xs, output_dims, name=None):
        assert xs.shape.to_integer_list == [params.batch_size,
                params.max_seq_len, params.num_units]
        name = name or 'rnn'
        seq_dim = xs.shape[1]
        xs = mtf.unstack(xs, seq_dim, name=f'{name}_unstack_op')

        if strategy == 0:
            layer_meshes, shapes = xs[0].mesh, xs[0].shape
            repl_fns = [None, lambda x: mtf.rename_dimension(x,
                x.shape[-1].name, shapes[-1].name)]

        elif strategy == 1:
            layer_meshes = [meshes[0], meshes[1]]
            shape = xs[0].shape
            new_shape = shape.rename_dimension('axis0', 'axis1')
            shapes = [shape, new_shape]
            repl_fns = [None, lambda x: ReplaceMeshWithDuplicates(x, meshes[1],
                name='replace_x_mesh')]
        else:
            layer_meshes, shapes = meshes, xs[0].shape
            repl_fns = [None, lambda x: ReplaceMesh(x, meshes[1],
                x.shape.rename_dimension(x.shape[-1].name, shapes[-1].name),
                name='replace_mesh_1')]

        rnn_stack = RNNStack(layer_meshes, shapes, repl_fns, output_dims,
                f'{name}_stack')
        ys = rnn_stack(xs)

        if strategy == 1:
            ys = [ReplaceMeshWithDuplicates(y, meshes[3],
                name=f'replace_ys_mesh_{i}') for i, y in enumerate(ys)]

        assert len(ys) == params.max_seq_len
        return mtf.stack(ys, seq_dim.name, 1, f'{name}_stack_op'), rnn_stack.hs


    # Model
    with tf.variable_scope('rnnlm'):
        embed = mtf.layers.embedding(mtf_src, vocab_dim, embed_dim, tf.float32,
                name='enc_embedding')
        out, _ = RNN(embed, rnn_out_dims, name='encoder_rnn')

    # Loss function
    with tf.variable_scope('loss'):
        if strategy == 1:
            new_dim_names = out.shape.dimension_names
            assert not new_dim_names[0].startswith('axis') \
                    and not new_dim_names[1].startswith('axis') \
                    and new_dim_names[2] == 'axis0'
            new_dim_names[2] = RandName()
            out = ReplaceMeshWithIndependentAxes(out, meshes[2], new_dim_names,
                    name='replace_out_mesh')
        elif strategy == 2:
            mtf_tgt = ReplaceMesh(mtf_tgt, meshes[1],
                    name='replace_mtf_tgt_mesh')

        out = mtf.layers.dense(out, final_proj_dim, use_bias=False,
                name='final_projection')
        one_hot_labels = mtf.one_hot(mtf_tgt, out.shape[-1], dtype=out.dtype)
        assert out.mesh == one_hot_labels.mesh
        assert out.shape == one_hot_labels.shape
        out = mtf.layers.softmax_cross_entropy_with_logits(out, one_hot_labels,
                out.shape[-1])
        loss = mtf.reduce_mean(out)

    with tf.variable_scope('optimize'):
        grads = mtf.gradients([loss], [v.outputs[0] for v in
            graph.trainable_variables])
        lr = 0.01
        opt = mtf.optimize.SgdOptimizer(lr)
        grad_updates = opt.apply_grads(grads, graph.trainable_variables)

    print(f'{datetime.datetime.now()} Lowering mtf ops...', flush=True)
    lowering = mtf.Lowering(graph, mesh_to_impl)
    print(f'{datetime.datetime.now()} Finished lowering.', flush=True)
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

    parser.add_argument('-b', '--batch', type=int, required=False, default=32,
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
                    2: Expert designed")
    parser.add_argument('--vocab', type=str, help="Source vocab data file.")
    parser.add_argument('--text', type=str, help="Source text data file.")
    args = parser.parse_args()
    params = Params(args.batch)
    [print(f'{arg} : {val}') for arg, val in vars(args).items()]

    if args.procs != 8:
        raise NotImplementedError('Current implementation only handles 8 GPUs.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(args.procs))[:-1]
            
    # Initialize dataset
    dataset = TextDataLoader(args.batch, args.vocab, None, args.text, None,
            max_seq_len=params.max_seq_len)
    inputs, labels, _, _ = dataset.next_batch()

    with open(args.vocab) as f:
        for vocab_size, _ in enumerate(f):
            pass
    vocab_size = int(math.ceil(vocab_size / 8)) * int(8)
    print("Vocab size: %d" % vocab_size)

    init_ops, loss_op, grad_ops = RNNLM(inputs, labels, params, vocab_size,
            args)
 
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


