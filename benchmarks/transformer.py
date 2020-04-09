import numpy as np
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf 

import math
from collections import namedtuple
import sys, time, os
import string, random
import argparse

import common
from dataloader import TextDataLoader
import utils
from utils import RandName
import mtf_operations as mt
import mesh_transformations as mesh_trans


class Params():
    def __init__(self, batch_size, max_seq_len):
        self.batch_size = batch_size
        self.nx = 6
        self.max_seq_len = max_seq_len
        self.d_model = 1024
        self.heads = 8
        self.d_ff = 4096 #self.heads * 512
        self.d_k = 128

def check_distribution(x, mesh, split_axes):
    assert x.mesh == mesh
    assert all(i < x.shape.ndims for i in split_axes)
    for i, name in enumerate(x.shape.dimension_names):
        try:
            assert (name == split_axes[i]), (
                    f'Name mismatch: {(name, split_axes[i])}')
        except KeyError:
            assert (not name.startswith('axis'))

def CreateMeshes(strategy, src, tgt, num_nodes, num_gpus, params):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}
    gpus_per_node = num_gpus // num_nodes

    mesh_id = 0
    def CreateMesh(mesh_shape):
        nonlocal mesh_id
        num_nodes = ((utils.Prod(mesh_shape) + gpus_per_node - 1) //
                gpus_per_node)

        mesh = mtf.Mesh(graph, f'mesh{mesh_id}')
        meshes.append(mesh)
        mesh_id += 1
        mesh_to_impl[mesh] = utils.GetMeshImpl(mesh_shape, num_nodes=num_nodes,
                gpus_per_node=gpus_per_node)
        return mesh

    if strategy == 0: # Data-parallel
        mesh = CreateMesh([num_gpus])
        shape = utils.ConvertToShape([('axis0', params.batch_size),
            params.max_seq_len])
        mtf_src = mtf.import_tf_tensor(mesh, src, shape)
        mtf_tgt = mtf.import_tf_tensor(mesh, src, shape)

    elif strategy == 1: # Opt strategy from the tool
        if num_gpus == 4:
            embed_dim1, embed_dim2 = 2, 2
            dim1, dim2 = 4, 1
        elif num_gpus == 8:
            embed_dim1, embed_dim2 = 2, 4
            dim1, dim2 = 8, 1
        elif num_gpus == 16:
            embed_dim1, embed_dim2 = 2, 8
            dim1, dim2 = 8, 2
        elif num_gpus == 32:
            embed_dim1, embed_dim2 = 4, 8
            dim1, dim2 = 16, 2
        elif num_gpus == 64:
            embed_dim1, embed_dim2 = 4, 16
            dim1, dim2 = 16, 4
        else:
            assert False
        assert ((embed_dim1 * embed_dim2) == num_gpus)
        assert ((dim1 * dim2) == num_gpus)

        mesh0 = CreateMesh([dim1, dim2])
        mesh1 = CreateMesh([embed_dim1, embed_dim2])

        shape = utils.ConvertToShape([params.batch_size,
            ('axis0', params.max_seq_len)])
        mtf_src = mtf.import_tf_tensor(mesh1, src, shape)
        mtf_tgt = mtf.import_tf_tensor(mesh1, tgt, shape)

    elif strategy == 2: # Strategy from mesh-tensorflow paper
        if num_gpus == 4:
            dim1, dim2 = 2, 2
        elif num_gpus == 8:
            dim1, dim2 = 2, 4
        elif num_gpus == 16:
            dim1, dim2 = 4, 4
        elif num_gpus == 32:
            dim1, dim2 = 4, 8
        elif num_gpus == 64:
            dim1, dim2 = 8, 8
        else:
            assert False
        assert ((dim1 * dim2) == num_gpus)

        mesh = CreateMesh([dim1, dim2])
        shape = utils.ConvertToShape([('axis0', params.batch_size),
            params.max_seq_len])
        mtf_src = mtf.import_tf_tensor(mesh, src, shape)
        mtf_tgt = mtf.import_tf_tensor(mesh, src, shape)
        
    else:
        assert False

    mtf_src = mtf.cast(mtf_src, tf.int32)
    mtf_tgt = mtf.cast(mtf_tgt, tf.int32)
    return graph, meshes, mesh_to_impl, mtf_src, mtf_tgt

def positional_encoding(x):
    seq_len_dim, model_dim = x.shape[1:]
    seq_len_size = seq_len_dim.size
    model_size = model_dim.size

    # mtf.constant is only to create a tensor with a constant scalar val. But as
    # long as the tensor is fully replicated, initializing it with a tensor
    # works.
    assert (not seq_len_dim.name.startswith('axis'))
    assert (not model_dim.name.startswith('axis'))

    # Values for positional encoder
    pos = np.arange(seq_len_size).reshape(-1, 1)
    val = np.power(10000, (2 * np.arange(model_size))/model_size, dtype=float)
    pos_enc_values = pos / val
    np.sin(pos_enc_values[:,::2], out=pos_enc_values[:,::2], dtype=np.float32)
    np.cos(pos_enc_values[:,1::2], out=pos_enc_values[:,1::2], dtype=np.float32)
    
    # positional encoder
    pos_enc = mtf.constant(x.mesh, pos_enc_values, shape=mtf.Shape([seq_len_dim,
        model_dim]), dtype=tf.float32)
    return (x * math.sqrt(model_size)) + pos_enc

def add_norm(x, y, name=None):
    assert x.mesh == y.mesh
    assert (x.shape == y.shape), (x.shape, y.shape)
    name = name or 'add_norm'
    z = mtf.add(x, y, output_shape=x.shape)
    return mtf.layers.layer_norm(z, dim=z.shape[-1], name=name)

def Transformer(src, tgt, params, src_vocab_size, tgt_vocab_size, strategy,
        num_nodes, num_gpus):
    graph, meshes, mesh_to_impl, mtf_src, mtf_tgt = CreateMeshes(
            strategy, src, tgt, num_nodes, num_gpus, params)
    src_vocab_size = utils.RoundUp(src_vocab_size, num_gpus)
    tgt_vocab_size = utils.RoundUp(tgt_vocab_size, num_gpus)

    # mtf dimensions
    if strategy == 0:
        src_vocab_dim  = mtf.Dimension(RandName(), src_vocab_size)
        tgt_vocab_dim  = mtf.Dimension(RandName(), tgt_vocab_size)
        model_dim      = mtf.Dimension(RandName(), params.d_model)
        d_k_dim        = mtf.Dimension(RandName(), params.d_k)
        heads_dim      = mtf.Dimension(RandName(), params.heads)
        ff_dim         = mtf.Dimension(RandName(), params.d_ff)
    elif strategy == 1:
        src_vocab_dim  = mtf.Dimension('axis1', src_vocab_size)
        tgt_vocab_dim  = mtf.Dimension('axis1', tgt_vocab_size)
        model_dim      = mtf.Dimension(RandName(), params.d_model)
        d_k_dim        = mtf.Dimension(RandName(), params.d_k)
        heads_dim      = mtf.Dimension('axis1', params.heads)
        ff_dim         = mtf.Dimension('axis1', params.d_ff)
    elif strategy == 2:
        src_vocab_dim  = mtf.Dimension('axis1', src_vocab_size)
        tgt_vocab_dim  = mtf.Dimension('axis1', tgt_vocab_size)
        model_dim      = mtf.Dimension(RandName(), params.d_model)
        d_k_dim        = mtf.Dimension(RandName(), params.d_k)
        heads_dim      = mtf.Dimension('axis1', params.heads)
        ff_dim         = mtf.Dimension('axis1', params.d_ff)
    else:
        assert False
    seq_len_dim = mtf_src.shape[-1]
    assert mtf_src.shape[-1] == mtf_tgt.shape[-1]

    if strategy == 1:
        check_distribution(mtf_src, meshes[1], {1:'axis0'})
        check_distribution(mtf_tgt, meshes[1], {1:'axis0'})
    else:
        check_distribution(mtf_src, meshes[0], {0:'axis0'})
        check_distribution(mtf_tgt, meshes[0], {0:'axis0'})

    def encoder_decoder(inp, enc_out, vocab_dim, name):
        with tf.variable_scope(name):
            # Embedding
            embed = mtf.layers.embedding(inp, vocab_dim, model_dim,
                    tf.float32, name=f'{name}_embedding')
            if strategy == 1:
                check_distribution(embed, meshes[1], {1:'axis0'})
                shape = embed.shape.rename_dimension('axis0', RandName())
                shape = shape.rename_dimension(shape[0].name, 'axis0')
                embed = mesh_trans.ReplaceMeshWithIndependentAxes(
                        embed, meshes[0], dim_names=shape.dimension_names)
            check_distribution(embed, meshes[0], {0:'axis0'})

            # Positional encoding
            x = positional_encoding(embed)
            check_distribution(x, meshes[0], {0:'axis0'})

            # Encoder/decoder layers
            for i in range(params.nx):
                # Multihead attention
                y = mtf.layers.multihead_attention(x, None, None, d_k_dim,
                        heads_dim, dropout=0.5, name=f'{name}_att_{i}')
                x = add_norm(x, y, name=f'{name}_att_{i}_norm')
                check_distribution(x, meshes[0], {0:'axis0'})

                if enc_out is not None:
                    y = mtf.layers.multihead_attention(x, enc_out, None,
                            d_k_dim, heads_dim, dropout=0.5,
                            name=f'{name}_att2_{i}')
                    x = add_norm(x, y, name=f'{name}_att2_{i}_norm')
                    check_distribution(x, meshes[0], {0:'axis0'})

                # Feed forward
                y = mtf.layers.dense_relu_dense(x, ff_dim, dropout=0.5,
                        name=f'{name}_ff_{i}')
                x = add_norm(x, y, name=f'{name}_ff_{i}_norm')
                check_distribution(x, meshes[0], {0:'axis0'})
            return x

    # Encoder/Decoder
    enc_out = encoder_decoder(mtf_src, None, src_vocab_dim, 'encoder')
    check_distribution(enc_out, meshes[0], {0:'axis0'})
    enc_out = mt.rename_dimension(enc_out, enc_out.shape[1].name, RandName())
    dec_out = encoder_decoder(mtf_tgt, enc_out, tgt_vocab_dim, 'decoder')

    # Loss function
    with tf.variable_scope('loss'):
        check_distribution(dec_out, meshes[0], {0:'axis0'})
        if strategy == 1:
            shape = dec_out.shape.rename_dimension('axis0',
                    mtf_tgt.shape[0].name)
            shape = shape.rename_dimension(shape[1].name, 'axis0')
            dec_out = mesh_trans.ReplaceMeshWithIndependentAxes(
                    dec_out, meshes[1], dim_names=shape.dimension_names)
            check_distribution(dec_out, meshes[1], {1:'axis0'})

        out = mtf.layers.dense(dec_out, tgt_vocab_dim, use_bias=False,
                reduced_dims=dec_out.shape[-1:], name='final_projection')
        assert out.mesh == mtf_tgt.mesh
        assert (out.shape.dims == mtf_tgt.shape.dims + [tgt_vocab_dim])
        out = mtf.layers.softmax_cross_entropy_with_logits(out, mtf_tgt,
                tgt_vocab_dim)
        loss = mtf.reduce_mean(out)

    return graph, mesh_to_impl, loss

def main():
    trainer = common.Trainer()
    args = trainer.args
    params = Params(args.batch_size, args.seq_len)

    # Initialize dataset
    dataset = TextDataLoader(args.batch_size, args.src_vocab, args.tgt_vocab,
            args.src_text, args.tgt_text, params.max_seq_len,
            args.src_vocab_size, args.tgt_vocab_size, args.sentences_size)
    enc_inputs, dec_inputs, _, _ = dataset.next_batch()

    # Model
    graph, mesh_to_impl, mtf_loss = Transformer(enc_inputs, dec_inputs, params,
            dataset.src_vocab_size, dataset.tgt_vocab_size, args.strategy,
            trainer.num_nodes, trainer.num_gpus)
 
    # Train
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    config = tf.ConfigProto(allow_soft_placement=False)
    trainer.train_model(graph, mesh_to_impl, mtf_loss, dataset, config=config,
            run_options=run_options)


if __name__ == '__main__':
    main()

