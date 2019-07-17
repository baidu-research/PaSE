import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf 

import math
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
        self.num_units = 1024
        self.max_seq_len = 256
        self.num_layers = 4


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

    else:
        assert False

    return graph, meshes, mesh_to_impl, mtf_src, mtf_tgt


def Seq2seq(src, tgt, params, src_vocab_size, tgt_vocab_size, args):
    strategy = args.strategy
    graph, meshes, mesh_to_impl, mtf_src, mtf_tgt = CreateMeshes(strategy, src,
            tgt, params)

    if strategy == 0:
        src_vocab_dim = mtf.Dimension(RandName(), src_vocab_size)
        tgt_vocab_dim = mtf.Dimension(RandName(), tgt_vocab_size)
        embed_dim = mtf.Dimension(RandName(), params.num_units)
        enc_rnn_output_dim = mtf.Dimension(RandName(), 4*params.num_units)
        enc_rnn_output_dims = [enc_rnn_output_dim] * params.num_layers
        dec_rnn_output_dims = enc_rnn_output_dims
        attn_dense_dim = mtf.Dimension(RandName(), params.num_units)
        final_proj_dim = tgt_vocab_dim

    elif strategy == 1:
        ...
    else:
        ...

    seq_len_dim = mtf_src.shape[-1]
    assert mtf_src.shape[-1] == mtf_tgt.shape[-1]

    class LSTMCell():
        def __init__(self, output_dim, h=None, name=None):
            self.name = name or 'lstm'
            self.output_dim = output_dim
            if h is not None:
                self.h = h

        def __call__(self, x, context=None):
            assert x.shape.to_integer_list == [params.batch_size, params.num_units]

            try:
                #assert x.shape[0] == self.h.shape[0]
                #if x.shape[1] != self.h.shape[1]:
                #    self.h = mtf.rename_dimension(self.h, self.h.shape[1].name,
                #            x.shape[1].name)
                assert x.shape == self.h.shape
                concat_tsrs = (x, self.h)
            except AttributeError:
                self.h = mtf.zeros(x.mesh, x.shape)
                concat_tsrs = (x, self.h)
            if context is not None:
                assert x.shape == context.shape
                concat_tsrs += context,
 
            xh = mtf.concat(concat_tsrs, x.shape[-1].name,
                    name=f'{self.name}_concat')
            if not hasattr(self, 'w'):
                self.w = mtf.get_variable(x.mesh, f'{self.name}_w',
                        mtf.Shape([xh.shape[-1], self.output_dim]))
            ifgo = mtf.einsum([xh, self.w], reduced_dims=[xh.shape[-1]],
                    name=f'{self.name}_ifgo')
            i, f, g, o = mtf.split(ifgo, ifgo.shape[-1], 4,
                    name=f'{self.name}_split_ifgo')

            i, f, o = (mtf.sigmoid(t) for t in (i, f, o))
            g = mtf.tanh(g)

            try:
                self.c = (f * self.c) + (i * g)
            except AttributeError:
                self.c = i * g
            self.h = o * mtf.tanh(self.c)
            self.h = mtf.rename_dimension(self.h, self.h.shape[-1].name,
                    x.shape[-1].name)
            return self.h

    class RNNStack():
        def __init__(self, output_dims, initial_hs, attention_fn=None, name=None):
            assert len(output_dims) == params.num_layers
            assert len(initial_hs) == params.num_layers

            self.name = name or 'rnn_stack'
            self.attention_fn = attention_fn
            self.lstms = []
            for i, (dim, h) in enumerate(zip(output_dims, initial_hs)):
                self.lstms.append(LSTMCell(dim, h, f'{self.name}_lstm_{i}'))

        def __call__(self, x):
            model_dim_name = x.shape[-1].name
            self.context = mtf.zeros(x.mesh, x.shape) \
                    if self.attention_fn is not None else None

            with tf.variable_scope(f'{self.name}_layer_{0}'):
                x = self.lstms[0](x, self.context)
            # Calculate context after 1st layer, to be used in next step
            context = self.attention_fn(x) \
                    if self.attention_fn is not None else None

            for i, lstm in enumerate(self.lstms[1:]):
                with tf.variable_scope(f'{self.name}_layer_{i+1}'):
                    #x = mtf.rename_dimension(x, x.shape[-1].name,
                    #        model_dim_name)
                    x = lstm(x, self.context)

            self.context = context # Update context
            return x

        @property
        def hs(self):
            return [lstm.h for lstm in self.lstms]

    def RNN(xs, output_dims, hs=None, attention_fn=None, name=None):
        assert xs.shape.to_integer_list == [params.batch_size,
                params.max_seq_len, params.num_units]
        name = name or 'rnn'
        seq_dim = xs.shape[1]
        hs = hs or [None]*params.num_layers
        xs = mtf.unstack(xs, seq_dim, name=f'{name}_unstack_op')

        rnn_stack = RNNStack(output_dims, hs, attention_fn, f'{name}_stack')
        ys = [rnn_stack(x) for x in xs]

        assert len(ys) == params.max_seq_len
        return mtf.stack(ys, seq_dim.name, 1, f'{name}_stack_op'), rnn_stack.hs

    # Encoder
    with tf.variable_scope('encoder'):
        embed = mtf.layers.embedding(mtf_src, src_vocab_dim, embed_dim,
                tf.float32, name='enc_embedding')
        enc_out, enc_hs = RNN(embed, enc_rnn_output_dims, name='encoder_rnn')

        # Encoder part of attention
        enc_attn = mtf.layers.dense(enc_out, attn_dense_dim, use_bias=False,
                name='enc_attention')

    # Decoder
    with tf.variable_scope('decoder'):
        embed = mtf.layers.embedding(mtf_tgt, tgt_vocab_dim, embed_dim,
                tf.float32, name='dec_embedding')
        enc_attn = mtf.rename_dimension(enc_attn, enc_attn.shape[-1].name,
                embed.shape[-1].name)

        def Attention(h):
            assert enc_attn.shape[-1] == h.shape[-1]
            score = mtf.einsum([enc_attn, h], reduced_dims=[enc_attn.shape[-1]],
                    name='attn_score_einsum')
            assert score.shape.to_integer_list == [params.batch_size,
                    params.max_seq_len]
            score = mtf.softmax(score, score.shape[-1], name=f'attn_score_softmax')
            context = mtf.einsum([score, enc_out],
                    reduced_dims=[score.shape[-1]], name='attn_context_einsum')
            return context

        dec_out, _ = RNN(embed, dec_rnn_output_dims, enc_hs,
                attention_fn=Attention, name='decoder_rnn')

    # Loss function
    with tf.variable_scope('loss'):
        out = mtf.layers.dense(dec_out, final_proj_dim, use_bias=False,
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

    init_ops, loss_op, grad_ops = Seq2seq(enc_inputs, dec_inputs, params,
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

