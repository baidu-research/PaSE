import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import sys, time, os
import string, random
import argparse

from dataloader import ImageDataLoader
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


def GetFilterShape(dims, dim_names):
    sh = mtf.Shape([mtf.Dimension(dim_names[0], dims[0]),
            mtf.Dimension(dim_names[1], dims[1]),
            mtf.Dimension(dim_names[2], dims[2]),
            mtf.Dimension(dim_names[3], dims[3])])
    return sh


def Concat(tsr_lst, name=None):
    assert all(tsr_lst[0].shape[:-1] == t.shape[:-1] for t in tsr_lst[1:])
    assert all(tsr_lst[0].shape[-1].name == t.shape[-1].name for t in
            tsr_lst[1:])
    return mtf.concat(tsr_lst, tsr_lst[0].shape[-1].name, name)


def CreateMeshes(strategy, img, labels, batch_size):
    h, w, ch = 299, 299, 3
    graph = mtf.Graph()
    meshes = {}

    if strategy == 0:
        mesh = mtf.Mesh(graph, 'mesh')
        meshes[mesh] = GetMeshImpl([8])

        mtf_img = mtf.import_tf_tensor(mesh, img, GetShape([('ma1', batch_size),
            h, w, ch]))
        mtf_labels = mtf.import_tf_tensor(mesh, labels, GetShape([('ma1',
            batch_size)]))

    elif strategy == 1:
        mesh1 = mtf.Mesh(graph, 'mesh1')
        meshes[mesh1] = GetMeshImpl([2])

        mesh2 = mtf.Mesh(graph, 'mesh2')
        meshes[mesh2] = GetMeshImpl([2, 4])

        mesh3 = mtf.Mesh(graph, 'mesh3')
        meshes[mesh3] = GetMeshImpl([4])

        mesh4 = mtf.Mesh(graph, 'mesh4')
        meshes[mesh4] = GetMeshImpl([2, 2])

        mesh5 = mtf.Mesh(graph, 'mesh5')
        meshes[mesh5] = GetMeshImpl([2, 2, 2])

        mtf_img = mtf.import_tf_tensor(mesh1, img, GetShape([('ma1',
            batch_size), h, w, ch]))
        mtf_labels = mtf.import_tf_tensor(mesh3, labels, GetShape([batch_size]))

    else:
        assert False

    return graph, meshes, mtf_img, mtf_labels


def AddBasicConv(img, fltr, stride=(1,1), padding='VALID', dim_name=None,
        rename_dim = True, name=None):
    dim_names = img.shape.dimension_names[1:] + [dim_name or RandName()]
    in_ch_dim_name = dim_names[-2]

    with tf.variable_scope(name, default_name='basic_conv'):
        conv = utils.Conv2d(img, GetFilterShape(fltr, dim_names), stride,
                padding)
        bn = mtf.layers.layer_norm(conv, conv.shape[0])
        if rename_dim:
            bn = mtf.rename_dimension(bn, bn.shape[-1].name, in_ch_dim_name)

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

        branch_pool = utils.MaxPool(img, (3, 3), padding='SAME')
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

        branch_pool = utils.MaxPool(img, (3, 3), stride=2)
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

        branch_pool = utils.AvgPool(img, (3, 3), padding='SAME')
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

        branch_pool = utils.MaxPool(img, (3, 3), stride=2)
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

        branch_pool = utils.AvgPool(img, (3, 3), stride=1, padding='SAME')
        branch_pool = AddBasicConv(branch_pool, (1, 1, in_channels, 192),
                dim_name=dim_name, name='branch_pool')
        return Concat([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                name='concat3')


def Inception(img, labels, args):
    num_classes = 1000
    graph, meshes, mtf_img, mtf_labels = CreateMeshes(args.strategy, img,
            labels, args.batch_size)

    with tf.variable_scope('inception'):
        conv1a = AddBasicConv(mtf_img, (3, 3, 3, 32), stride=2, name='conv1a')
        conv2a = AddBasicConv(conv1a, (3, 3, 32, 32), name='conv2a')
        conv2b = AddBasicConv(conv2a, (3, 3, 32, 64), padding='SAME',
                name='conv2b')
        pool = utils.MaxPool(conv2b, (3, 3), stride=2, name='pool1')

        conv3b = AddBasicConv(pool, (1, 1, 64, 80), name='conv3b')
        conv4a = AddBasicConv(conv3b, (3, 3, 80, 192), name='conv4a')
        pool = utils.MaxPool(conv4a, (3, 3), stride=2, name='pool2')

        mixed5b = AddInceptionA(pool, 192, 32, name='mixed5b')
        mixed5c = AddInceptionA(mixed5b, 256, 64, name='mixed5c')
        mixed5d = AddInceptionA(mixed5c, 288, 64, name='mixed5d')
        mixed6a = AddInceptionB(mixed5d, 288, name='mixed6a')
        mixed6b = AddInceptionC(mixed6a, 768, 128, name='mixed6b')
        mixed6c = AddInceptionC(mixed6b, 768, 160, name='mixed6c')
        mixed6d = AddInceptionC(mixed6c, 768, 160, name='mixed6d')
        mixed6e = AddInceptionC(mixed6d, 768, 192, name='mixed6e')
        mixed7a = AddInceptionD(mixed6e, 768, name='mixed7a')
        mixed7b = AddInceptionE(mixed7a, 1280, name='mixed7b')
        mixed7c = AddInceptionE(mixed7b, 2048, name='mixed7c')
        mean = mtf.reduce_mean(mixed7c, output_shape =
                mtf.Shape([mixed7c.shape[0], mixed7c.shape[-1]]))
        fc = mtf.layers.dense(mean, mtf.Dimension(RandName(), num_classes))

        with tf.variable_scope('loss'):
            one_hot_labels = mtf.one_hot(mtf_labels, fc.shape[-1])
            cross_ent = mtf.layers.softmax_cross_entropy_with_logits(fc,
                    one_hot_labels, fc.shape[-1])
            loss = mtf.reduce_mean(cross_ent)

        with tf.variable_scope('optimize'):
            grads = mtf.gradients([loss], [v.outputs[0] for v in
                graph.trainable_variables])
            opt = mtf.optimize.SgdOptimizer(0.01)
            grad_updates = opt.apply_grads(grads, graph.trainable_variables)

        print('Beginning to lower mtf graph...')
        lowering = mtf.Lowering(graph, meshes)
        print('Finished lowering.')
        tf_loss = lowering.export_to_tf_tensor(loss)
        tf_grad_updates = [lowering.lowered_operation(op) for op in grad_updates]

        # Initializer
        tf_init_vars = \
                utils.FlattenList([lowering.variables[var].laid_out_tensor.all_slices
                    for var in graph.all_variables])
        init_op = []
        for v in tf_init_vars:
            with tf.device(v.device):
                init_op.append(v.initializer)

        return init_op, [tf_loss] + tf_grad_updates


def main():
    parser = argparse.ArgumentParser(formatter_class =
            argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=256,
            help="Batch size.")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors.")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=10,
            help="No. of epochs")
    parser.add_argument('--display_steps', type=int, required=False, default=10,
            help="Steps to pass before displaying intermediate results")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(2)), 
            help="Strategy to use. 0: DataParallel, 1: Optimized.")
    parser.add_argument('--dataset_dir', type=str, required=False, default=None,
            help='Dataset directory')
    parser.add_argument('--labels_filename', type=str, required=False,
            default='labels.txt', help='Labels filename')
    parser.add_argument('--dataset_size', type=int, required=False,
            default=1000, help='Labels filename')
    args = parser.parse_args()

    if args.procs != 8:
        raise NotImplementedError('Current implementation only handles 8 GPUs.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(args.procs))[:-1]
    
    # Initalize the data generator seperately for the training and validation set
    dataset = ImageDataLoader(args.batch_size, (299, 299), dataset_size =
            args.dataset_size, dataset_dir=args.dataset_dir,
            labels_filename=args.labels_filename)
    train_batches_per_epoch = np.floor(dataset.dataset_size / args.batch_size).astype(np.int16)
    assert train_batches_per_epoch > 0
    
    # Input tensors
    tf_x, tf_y = dataset.next_batch()
    tf_x.set_shape([args.batch_size, 299, 299, 3])
    tf_y.set_shape([args.batch_size])

    init_ops, ops = Inception(tf_x, tf_y, args)

    config = tf.ConfigProto()
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.variable_scope('train'):
        with tf.Session(config=config) as sess:
            dataset.reset_pointer()
            sess.run(init_ops)

            tot_time = float(0)
            start = time.time()
            for epoch in range(args.epochs):
                step = 0

                for _ in range(train_batches_per_epoch):
                    loss_val, *rem = sess.run(ops)
                    step += 1

                    if step % args.display_steps == 0 and step > 0:
                        print("Epoch: " + str(epoch) + "; Loss: " +
                                str(loss_val))

                dataset.reset_pointer()
            end = time.time()
            tot_time += (end - start)

    img_per_sec = (dataset.dataset_size * args.epochs) / tot_time
    print("Throughout: " + str(img_per_sec) + " images / sec")


if __name__ == '__main__':
    main()

