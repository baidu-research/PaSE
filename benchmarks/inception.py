import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import sys
import time
import os
from argparse import ArgumentParser

from dataloader import ImageDataLoader
import utils


def AssignLayout(ta_axes, mesh_axis):
    layout = []
    for a in ta_axes:
        layout.append((a, mesh_axis))
    return layout


def Concat(tsr_lst, name=None):
    assert all(tsr_lst[0].shape[0] == t.shape[0] for t in tsr_lst[1:])
    assert all(tsr_lst[0].shape[1].name == t.shape[1].name for t in
            tsr_lst[1:])
    assert all(tsr_lst[0].shape[2:] == t.shape[2:] for t in tsr_lst[1:])
    return mtf.concat(tsr_lst, tsr_lst[0].shape[1], name)


def GetFilterShape(dims, dim_names):
    sh = mtf.Shape(mtf.Dimension(dim_names[0], dims[0]),
            mtf.Dimension(dim_names[1], dims[1]),
            mtf.Dimension(dim_names[2], dims[2]),
            mtf.Dimension(dim_names[3], dims[3]))
    return sh


def AddBasicConv(img, fltr, stride=(1,1), padding='VALID', dim_name=None,
        rename_dim = True, name=None):
    dim_names = img.shape.dimension_names[1:] + [dim_name or 'n_dim']
    in_ch_dim_name = dim_names[-2]

    with tf.variable_scope(name, default_name='basic_conv'):
        conv = utils.Conv2d(img, GetFilterShape(fltr, dim_names), stride,
                padding)
        bn = mtf.layers.batch_norm(conv, True, momentum=0.99)
        if rename_dim:
            bn = mtf.rename_dimension(bn, x.shape[-1].name, in_ch_dim_name)

        return bn


def AddInceptionA(img, in_channels, pool_features, dim_name=None, name=None):
    with tf.variable_scope(name, default_name='inception_A'):
        branch1x1 = AddBasicConv(img, ((1, 1, in_channels, 64)),
                dim_name=dim_name, name='branch1x1')

        branch5x5 = AddBasicConv(img, ((1, 1, in_channels, 48)),
                dim_name=dim_name, name='branch5x5_1')
        branch5x5 = AddBasicConv(branch5x5, ((5, 5, 48, 64)),
                padding='SAME', dim_name=dim_name, name='branch5x5_2')

        branch3x3dbl = AddBasicConv(img, ((1, 1, in_channels,
            64)), dim_name=dim_name, name='branch3x3_1')
        branch3x3dbl = AddBasicConv(branch3x3dbl, ((3, 3, 64,
            96)), padding='SAME', dim_name=dim_name, name='branch3x3_2')
        branch3x3dbl = AddBasicConv(branch3x3dbl, ((3, 3, 96,
            96)), padding='SAME', dim_name=dim_name, name='branch3x3_3')

        branch_pool = utils.Pooling(img, (3, 3), padding='SAME')
        branch_pool = AddBasicConv(branch_pool, ((1, 1,
            in_channels, pool_features)), dim_name=dim_name, name='branch_pool')
        return Concat([branch1x1, branch5x5, branch3x3dbl, branch_pool])


def AddInceptionB(img, in_channels, dim_name=None, name=None):
    with tf.variable_scope(name, default_name='inception_B'):
        branch3x3 = AddBasicConv(img, ((3, 3, in_channels, 384)),
                stride=2, dim_name=dim_name, name='branch3x3')

        branch3x3dbl = AddBasicConv(img, ((1, 1, in_channels,
            64)), dim_name=dim_name, name='branch3x3_1')
        branch3x3dbl = AddBasicConv(branch3x3dbl, ((3, 3, 64,
            96)), padding='SAME', dim_name=dim_name, name='branch3x3_2')
        branch3x3dbl = AddBasicConv(branch3x3dbl, ((3, 3, 96,
            96)), stride=2, dim_name=dim_name, name='branch3x3_3')

        branch_pool = utils.Pooling(img, (3, 3), stride=2)
        return Concat([branch3x3, branch3x3dbl, branch_pool])


def AddInceptionC(img, in_channels, channels_7x7, dim_name=None, name=None):
    with tf.variable_scope(name, default_name='inception_C'):
        branch1x1 = AddBasicConv(img, ((1, 1, in_channels, 192)),
                dim_name=dim_name, name='branch1x1')

        branch7x7 = AddBasicConv(img, ((1, 1, in_channels, channels_7x7)),
                dim_name=dim_name, name='branch7x7_1')
        branch7x7 = AddBasicConv(branch7x7, ((1, 7, channels_7x7,
            channels_7x7)), padding='SAME', dim_name=dim_name,
            name='branch7x7_2')
        branch7x7 = AddBasicConv(branch7x7, ((7, 1, channels_7x7, 192)),
                padding='SAME', dim_name=dim_name, name='branch7x7_3')

        branch7x7_dbl = AddBasicConv(img, ((1, 1, in_channels, channels_7x7)),
                dim_name=dim_name, name='branch7x7_dbl_1')
        branch7x7_dbl = AddBasicConv(branch7x7_dbl, ((7, 1, channels_7x7,
            channels_7x7)), padding='SAME', dim_name=dim_name,
            name='branch7x7_dbl_2')
        branch7x7_dbl = AddBasicConv(branch7x7_dbl, ((1, 7, channels_7x7,
            channels_7x7)), padding='SAME', dim_name=dim_name,
            name='branch7x7_dbl_3')
        branch7x7_dbl = AddBasicConv(branch7x7_dbl, ((7, 1, channels_7x7,
            channels_7x7)), padding='SAME', dim_name=dim_name,
            name='branch7x7_dbl_4')
        branch7x7_dbl = AddBasicConv(branch7x7_dbl, ((1, 7, channels_7x7, 192)),
                padding='SAME', dim_name=dim_name, name='branch7x7_dbl_5')

        branch_pool = utils.Pooling(img, (3, 3), stride=1, pad='SAME')
        branch_pool = AddBasicConv(branch_pool, ((1, 1, in_channels, 192)),
                dim_name=dim_name, name='branch_pool')
        return Concat([branch1x1, branch7x7, branch7x7_dbl, branch_pool])


def AddInceptionD(img, in_channels, dim_name=None, name=None):
    with tf.variable_scope(name, default_name='inception_D'):
        branch3x3 = AddBasicConv(img, ((1, 1, in_channels, 192)),
                dim_name=dim_name, name='branch3x3_1')
        branch3x3 = AddBasicConv(branch3x3, ((3, 3, 192, 320)), stride=2,
                dim_name=dim_name, name='branch3x3_2')

        branch7x7x3 = AddBasicConv(img, ((1, 1, in_channels, 192)),
                dim_name=dim_name, name='branch7x7x3_1')
        branch7x7x3 = AddBasicConv(branch7x7x3, ((1, 7, 192, 192)),
                padding='SAME', dim_name=dim_name, name='branch7x7x3_2')
        branch7x7x3 = AddBasicConv(branch7x7x3, ((7, 1, 192, 192)),
                padding='SAME', dim_name=dim_name, name='branch7x7x3_3')
        branch7x7x3 = AddBasicConv(branch7x7x3, ((3, 3, 192, 192)),
                stride=2, dim_name=dim_name, name='branch7x7x3_4')

        branch_pool = utils.Pooling(img, (3, 3), stride=2)
        return Concat([branch3x3, branch7x7x3, branch_pool])


def AddInceptionE(img, in_channels, dim_name=None, name=None):
    with tf.variable_scope(name, default_name='inception_E'):
        branch1x1 = AddBasicConv(img, (1, 1, in_channels, 320),
                dim_name=dim_name, name='branch1x1')

        branch3x3 = AddBasicConv(img, (1, 1, in_channels, 384),
                dim_name=dim_name, name='branch3x3')
        branch3x3_2a = AddBasicConv(branch3x3, (1, 3, 384, 384), padding='SAME',
                dim_name=dim_name, name='branch3x3_2a')
        branch3x3_2b = AddBasicConv(branch3x3, (3, 1, 384, 384), padding='SAME',
                dim_name=dim_name, name='branch3x3_2b')
        branch3x3 = Concat([branch3x3_2a, branch3x3_2b], name='concat1')

        branch3x3dbl = AddBasicConv(img, (1, 1, in_channels, 448),
                dim_name=dim_name, name='branch3x3dbl')
        branch3x3dbl = AddBasicConv(branch3x3dbl, (3, 3, 448, 384),
                padding='SAME', dim_name=dim_name, name='branch3x3dbl_1')
        branch3x3dbl_3a = AddBasicConv(branch3x3dbl, (1, 3, 384, 384),
                padding='SAME', dim_name=dim_name, name='branch3x3dbl_3a')
        branch3x3dbl_3b = AddBasicConv(branch3x3dbl, (3, 1, 384, 384),
                padding='SAME', dim_name=dim_name, name='branch3x3dbl_3b')
        branch3x3dbl = Concat([branch3x3dbl_3a, branch3x3dbl_3b], name='concat2')

        branch_pool = utils.Pooling(img, (3, 3), stride=1, padding='SAME')
        branch_pool = AddBasicConv(branch_pool, (1, 1, in_channels, 192),
                dim_name=dim_name, name='branch_pool')
        return Concat([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                name='concat3')


def Inception(img, labels, args):
    num_classes = 1000

    # mtf dimensions
    conv_batch_dim = mtf.Dimension('conv_batch_dim', args.batch_size)
    fc_batch_dim = mtf.Dimension('fc_batch_dim', args.batch_size)
    c_dim = mtf.Dimension('c_dim', 3)
    h_dim = mtf.Dimension('h_dim', 299)
    w_dim = mtf.Dimension('w_dim', 299)

    graph = mtf.Graph()
    if args.strategy == 0:
        mesh = mtf.Mesh(graph, 'mesh1')
        mesh_shape = [('p1', args.procs)]
        devices = ['gpu:%d' %i for i in range(args.procs)]
        layout = AssignLayout([conv_batch_dim.name, fc_batch_dim.name], 'p1')
        mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape,
                layout, devices)

        meshes = {mesh:mesh_impl}

        mtf_x = mtf.import_tf_tensor(mesh, tf_x, mtf.Shape([conv_batch_dim,
            h_dim, w_dim, c_dim]))
        mtf_y = mtf.import_tf_tensor(mesh, tf_y, mtf.Shape([fc_batch_dim]))
    else:
        assert False

    with tf.variable_scope('inception'):
        conv1a = AddBasicConv(img, (3, 3, 3, 32), stride=2, name='conv1a')
        conv2a = AddBasicConv(conv1a, (3, 3, 32, 32), name='conv2a')
        conv2b = AddBasicConv(conv2a, (3, 3, 32, 64), padding='SAME',
                name='conv2b')
        pool = utils.Pooling(conv2b, (3, 3), stride=2, name='pool1')

        conv3b = AddBasicConv(pool, (1, 1, 64, 80), name='conv3b')
        conv4a = AddBasicConv(conv3b, (3, 3, 80, 192), name='conv4a')
        pool = utils.Pooling(conv4a, (3, 3), stride=2, name='pool2')

        mixed5b = AddInceptionA(pool, 192, 32, name='mixed5b')
        mixed5c = AddInceptionA(mixed5b, 256, 64, name='mixed5c')
        mixed5d = AddInceptionA(mixed5c, 288, 64, name='mixed5d')
        mixed6a = AddInceptionB(mixed5d, 288, name='mixed6a')
        mixed6b = AddInceptionC(mixed6a, 768, 128)
        mixed6c = AddInceptionC(mixed6b, 768, 160)
        mixed6d = AddInceptionC(mixed6c, 768, 160)
        mixed6e = AddInceptionC(mixed6d, 768, 192)
        mixed7a = AddInceptionD(mixed6e, 768)
        mixed7b = AddInceptionE(mixed7a, 1280)
        mixed7c = AddInceptionE(mixed7b, 2048)
        mean = mtf.reduce_mean(mixed7c, reduced_dim=mixed3c.shape[1:3])
        fc = mtf.dense(mean, num_classes_dim)

        with tf.variable_scope('loss'):
            one_hot_labels = mtf.one_hot(labels, fc.shape[-1])
            cross_ent = mtf.layers.softmax_cross_entropy_with_logits(fc,
                    one_hot_labels, fc.shape[-1])
            loss = mtf.reduce_mean(cross_ent)

        with tf.variable_scope('optimize'):
            grads = mtf.gradients([loss], [v.outputs[0] for v in
                graph.trainable_variables])
            opt = mtf.optimize.SgdOptimizer(0.01)
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

        return init_op, [tf_loss] + tf_grad_updates


def main():
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=256,
            help="Batch size. (Default: 256)")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors. (Default: 8)")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
            help="No. of epochs")
    parser.add_argument('--display_steps', type=int, required=False, default=10,
            help="Steps to pass before displaying intermediate results")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(3)), 
            help="Strategy to use. 0: DataParallel, \
                    1: Optimized for 1080Ti, \
                    2: Optimized for DGX. \
                    (Default: 0) ")
    parser.add_argument('--dataset_dir', type=str, required=False, default=None,
            help='Dataset directory')
    parser.add_argument('--labels_filename', type=str, required=False,
            default=None, help='Labels filename')
    parser.add_argument('--dataset_size', type=int, required=False,
            default=1000, help='Labels filename')
    args = parser.parse_args()

    if args.procs != 8:
        raise NotImplementedError('Current implementation only handles 8 GPUs.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(args.procs))[:-1]
    
    # Initalize the data generator seperately for the training and validation set
    dataset_dir = args.dataset_dir
    labels_filename = args.labels_filename
    dataset_size = args.dataset_size
    if dataset_dir is None or labels_filename is None:
        dataset = ImageDataLoader(args.batch_size, dataset_size=dataset_size,
                synthetic=True)
    else:
        dataset = ImageDataLoader(args.batch_size, dataset_dir=dataset_dir,
                labels_filename=labels_filename, synthetic=False)
    train_batches_per_epoch = np.floor(dataset.dataset_size / args.batch_size).astype(np.int16)
    assert train_batches_per_epoch > 0
    
    # Input tensors
    tf_x, tf_y = dataset.next_batch()
    tf_x.set_shape([args.batch_size, 299, 299, 3])
    tf_y.set_shape([args.batch_size])

    init_ops, ops = Inception(tf_x, tf_y, args)

    with tf.variable_scope('train'):
        with tf.Session as sess:
            dataset.reset_pointer()
            sess.run(init_ops)

            tot_time = float(0)
            start = time.time()
            for epoch in range(args.epochs):
                for step in range(train_batches_per_epoch):
                    loss_val, *rem = sess.run([tf_loss] + tf_grad_updates)

                if epoch % args.display_steps == 0 and epoch > 0:
                    print("Epoch: " + str(epoch) + "; Loss: " + str(loss_val))

                dataset.reset_pointer()
            end = time.time()
            tot_time += (end - start)

    img_per_sec = (dataset.dataset_size * num_epochs) / tot_time
    print("Throughout: " + str(img_per_sec) + " images / sec")


if __name__ == '__main__':
    main()
