import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import datetime
import sys, time, os
import string, random
import argparse

from dataloader import ImageDataLoader
from utils import FlattenList, GetMeshImpl
from mtf_operations import Conv2d, MaxPool, AvgPool
from mesh_transformations import ReplaceMeshWithDuplicates, \
        ReplaceMeshWithConcatSplit

log_distribution = False


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


def Concat(tsr_lst, name=None):
    assert all(tsr_lst[0].shape[:-1] == t.shape[:-1] for t in tsr_lst[1:])

    concat_dim_name = RandName()
    concat_tsrs = []
    for t in tsr_lst:
        assert not t.shape[-1].name.startswith('axis')
        t = mtf.rename_dimension(t, t.shape[-1].name, concat_dim_name)
        concat_tsrs.append(t)

    return mtf.concat(concat_tsrs, concat_dim_name, name)


def CreateMeshes(strategy, img, labels, batch_size):
    h, w, ch = 299, 299, 3
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}

    def Mesh():
        mesh = mtf.Mesh(graph, 'mesh%d' % Mesh.idx)
        meshes.append(mesh)
        Mesh.idx += 1
        return mesh
    Mesh.idx = 0

    if strategy == 0:
        mesh = Mesh()
        mesh_to_impl[mesh] = GetMeshImpl([8])

        mtf_img = mtf.import_tf_tensor(mesh, img, GetShape([('axis0', batch_size),
            h, w, ch]))
        mtf_labels = mtf.import_tf_tensor(mesh, labels, GetShape([('axis0',
            batch_size)]))

    elif strategy == 1:
        # mesh0
        mesh = Mesh()
        mesh_to_impl[mesh] = GetMeshImpl([8])

        # mesh1
        mesh = Mesh()
        mesh_to_impl[mesh] = GetMeshImpl([4, 2])

        # mesh2
        mesh = Mesh()
        mesh_to_impl[mesh] = GetMeshImpl([4], [0, 2, 4, 6])

        mtf_img = mtf.import_tf_tensor(meshes[0], img, GetShape([('axis0',
            batch_size), h, w, ch]))
        mtf_labels = mtf.import_tf_tensor(meshes[1], labels, GetShape([batch_size]))

    else:
        # mesh0
        mesh = Mesh()
        mesh_to_impl[mesh] = GetMeshImpl([8])

        mtf_img = mtf.import_tf_tensor(meshes[0], img, GetShape([('axis0',
            batch_size), h, w, ch]))
        mtf_labels = mtf.import_tf_tensor(meshes[0], labels, GetShape([batch_size]))

    return graph, meshes, mesh_to_impl, mtf_img, mtf_labels


def BasicConv(img, fltr, stride=(1,1), padding='VALID', dim_name=None,
        rename_dim=False, name=None):
    if dim_name is None or dim_name in img.shape.dimension_names:
        assert dim_name is None or not dim_name.startswith('axis')
        dim_name = RandName()
    dim_names = img.shape.dimension_names[1:] + [dim_name]
    in_ch_dim_name = dim_names[-2]

    filter_shape = lambda x, y: mtf.Shape([mtf.Dimension(name, size) for name,
        size in zip(x, y)])
    with tf.variable_scope(name, default_name='basic_conv'):
        conv = Conv2d(img, filter_shape(dim_names, fltr), stride, padding)
        bn = mtf.layers.layer_norm(conv, conv.shape[0])
        if rename_dim:
            bn = mtf.rename_dimension(bn, bn.shape[-1].name, in_ch_dim_name)

        def LogDistribution(tsr, end='\n'):
            if log_distribution:
                print(tsr.shape)
                mesh_impl = mesh_to_impl[tsr.mesh]
                print('(', end='')
                for d in tsr.shape.dims:
                    print(mtf.tensor_dim_to_mesh_dim_size(mesh_impl.layout_rules,
                        mesh_impl.shape, d), end=', ')
                print(')', end=end, flush=True)

        LogDistribution(img)
        LogDistribution(bn, '\n\n')

        return bn


def InceptionA(img, in_channels, pool_features, name=None):
    with tf.variable_scope(name, default_name='inception_A'):
        branch1x1 = BasicConv(img, ((1, 1, in_channels, 64)), name='branch1x1')

        branch5x5 = BasicConv(img, ((1, 1, in_channels, 48)),
                name='branch5x5_1')
        branch5x5 = BasicConv(branch5x5, ((5, 5, 48, 64)), padding='SAME',
                name='branch5x5_2')

        branch3x3dbl = BasicConv(img, ((1, 1, in_channels, 64)),
                name='branch3x3_1')
        branch3x3dbl = BasicConv(branch3x3dbl, ((3, 3, 64, 96)), padding='SAME',
                name='branch3x3_2')
        branch3x3dbl = BasicConv(branch3x3dbl, ((3, 3, 96, 96)), padding='SAME',
                name='branch3x3_3')

        branch_pool = MaxPool(img, (3, 3), padding='SAME')
        branch_pool = BasicConv(branch_pool, ((1, 1, in_channels,
            pool_features)), name='branch_pool')

        y = Concat([branch1x1, branch5x5, branch3x3dbl, branch_pool])
        assert y.shape[0].name == 'axis0' \
                and not y.shape[1].name.startswith('axis') \
                and not y.shape[2].name.startswith('axis') \
                and not y.shape[3].name.startswith('axis')
        return y


def InceptionB(img, in_channels, name=None):
    with tf.variable_scope(name, default_name='inception_B'):
        branch3x3 = BasicConv(img, ((3, 3, in_channels, 384)), stride=2,
                name='branch3x3')

        branch3x3dbl = BasicConv(img, ((1, 1, in_channels, 64)),
                name='branch3x3_1')
        branch3x3dbl = BasicConv(branch3x3dbl, ((3, 3, 64, 96)), padding='SAME',
                name='branch3x3_2')
        branch3x3dbl = BasicConv(branch3x3dbl, ((3, 3, 96, 96)), stride=2,
                name='branch3x3_3')

        branch_pool = MaxPool(img, (3, 3), stride=2)
        y = Concat([branch3x3, branch3x3dbl, branch_pool])
        assert y.shape[0].name == 'axis0' \
                and not y.shape[1].name.startswith('axis') \
                and not y.shape[2].name.startswith('axis') \
                and not y.shape[3].name.startswith('axis')
        return y


def InceptionC(img, in_channels, channels_7x7, name=None):
    with tf.variable_scope(name, default_name='inception_C'):
        branch1x1 = BasicConv(img, ((1, 1, in_channels, 192)), name='branch1x1')

        branch7x7 = BasicConv(img, ((1, 1, in_channels, channels_7x7)),
                name='branch7x7_1')
        branch7x7 = BasicConv(branch7x7, ((1, 7, channels_7x7, channels_7x7)),
                padding='SAME', name='branch7x7_2')
        branch7x7 = BasicConv(branch7x7, ((7, 1, channels_7x7, 192)),
                padding='SAME', name='branch7x7_3')

        branch7x7_dbl = BasicConv(img, ((1, 1, in_channels, channels_7x7)),
                name='branch7x7_dbl_1')
        branch7x7_dbl = BasicConv(branch7x7_dbl, ((7, 1, channels_7x7,
            channels_7x7)), padding='SAME', name='branch7x7_dbl_2')
        branch7x7_dbl = BasicConv(branch7x7_dbl, ((1, 7, channels_7x7,
            channels_7x7)), padding='SAME', name='branch7x7_dbl_3')
        branch7x7_dbl = BasicConv(branch7x7_dbl, ((7, 1, channels_7x7,
            channels_7x7)), padding='SAME', name='branch7x7_dbl_4')
        branch7x7_dbl = BasicConv(branch7x7_dbl, ((1, 7, channels_7x7, 192)),
                padding='SAME', name='branch7x7_dbl_5')

        branch_pool = AvgPool(img, (3, 3), padding='SAME')
        branch_pool = BasicConv(branch_pool, ((1, 1, in_channels, 192)),
                name='branch_pool')
        y = Concat([branch1x1, branch7x7, branch7x7_dbl, branch_pool])
        assert y.shape[0].name == 'axis0' \
                and not y.shape[1].name.startswith('axis') \
                and not y.shape[2].name.startswith('axis') \
                and not y.shape[3].name.startswith('axis')
        return y


def InceptionD(img, in_channels, name=None):
    with tf.variable_scope(name, default_name='inception_D'):
        branch3x3 = BasicConv(img, ((1, 1, in_channels, 192)),
                name='branch3x3_1')
        branch3x3 = BasicConv(branch3x3, ((3, 3, 192, 320)), stride=2,
                name='branch3x3_2')

        branch7x7x3 = BasicConv(img, ((1, 1, in_channels, 192)),
                name='branch7x7x3_1')
        branch7x7x3 = BasicConv(branch7x7x3, ((1, 7, 192, 192)), padding='SAME',
                name='branch7x7x3_2')
        branch7x7x3 = BasicConv(branch7x7x3, ((7, 1, 192, 192)), padding='SAME',
                name='branch7x7x3_3')
        branch7x7x3 = BasicConv(branch7x7x3, ((3, 3, 192, 192)), stride=2,
                name='branch7x7x3_4')

        branch_pool = MaxPool(img, (3, 3), stride=2)
        y = Concat([branch3x3, branch7x7x3, branch_pool])
        assert y.shape[0].name == 'axis0' \
                and not y.shape[1].name.startswith('axis') \
                and not y.shape[2].name.startswith('axis') \
                and not y.shape[3].name.startswith('axis')
        return y


def InceptionE(img, in_channels, strategy, meshes=None, name=None):
    with tf.variable_scope(name, default_name='inception_E'):
        if strategy == 1:
            dim_name = 'axis1'
            rename_dim = True
            if meshes is not None:
                img = ReplaceMeshWithConcatSplit(img, meshes[2])
                img = ReplaceMeshWithDuplicates(img, meshes[1])
        else:
            dim_name = None
            rename_dim = False

        if strategy == 1:
            assert img.shape[0].name == 'axis0' \
                    and all(not name.startswith('axis') for name in
                            img.shape.dimension_names[1:])

        branch1x1 = BasicConv(img, (1, 1, in_channels, 320), dim_name=dim_name,
                rename_dim=rename_dim, name='branch1x1')

        branch3x3 = BasicConv(img, (1, 1, in_channels, 384), dim_name=dim_name,
                rename_dim=rename_dim, name='branch3x3')
        branch3x3_2a = BasicConv(branch3x3, (1, 3, 384, 384), padding='SAME',
                dim_name=dim_name, rename_dim=rename_dim, name='branch3x3_2a')
        branch3x3_2b = BasicConv(branch3x3, (3, 1, 384, 384), padding='SAME',
                dim_name=dim_name, rename_dim=rename_dim, name='branch3x3_2b')
        branch3x3 = Concat([branch3x3_2a, branch3x3_2b], name='concat1')

        branch3x3dbl = BasicConv(img, (1, 1, in_channels, 448),
                dim_name=dim_name, rename_dim=rename_dim, name='branch3x3dbl')
        branch3x3dbl = BasicConv(branch3x3dbl, (3, 3, 448, 384), padding='SAME',
                dim_name=dim_name, rename_dim=rename_dim, name='branch3x3dbl_1')
        branch3x3dbl_3a = BasicConv(branch3x3dbl, (1, 3, 384, 384),
                padding='SAME', dim_name=dim_name, rename_dim=rename_dim,
                name='branch3x3dbl_3a')
        branch3x3dbl_3b = BasicConv(branch3x3dbl, (3, 1, 384, 384),
                padding='SAME', dim_name=dim_name, rename_dim=rename_dim,
                name='branch3x3dbl_3b')
        branch3x3dbl = Concat([branch3x3dbl_3a, branch3x3dbl_3b], name='concat2')

        branch_pool = AvgPool(img, (3, 3), stride=1, padding='SAME')
        branch_pool = BasicConv(branch_pool, (1, 1, in_channels, 192),
                dim_name=dim_name, rename_dim=rename_dim, name='branch_pool')
        return Concat([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                name='concat3')


def Inception(img, labels, args):
    num_classes = 1000
    graph, meshes, mesh_to_impl, mtf_img, mtf_labels = \
            CreateMeshes(args.strategy, img, labels, args.batch_size)

    strategy = args.strategy
    with tf.variable_scope('inception'):
        conv1a = BasicConv(mtf_img, (3, 3, 3, 32), stride=2, name='conv1a')
        conv2a = BasicConv(conv1a, (3, 3, 32, 32), name='conv2a')
        conv2b = BasicConv(conv2a, (3, 3, 32, 64), padding='SAME',
                name='conv2b')
        pool = MaxPool(conv2b, (3, 3), stride=2, name='pool1')
        conv3b = BasicConv(pool, (1, 1, 64, 80), name='conv3b')
        conv4a = BasicConv(conv3b, (3, 3, 80, 192), name='conv4a')
        pool = MaxPool(conv4a, (3, 3), stride=2, name='pool2')

        mixed5b = InceptionA(pool, 192, 32, name='mixed5b')
        mixed5c = InceptionA(mixed5b, 256, 64, name='mixed5c')
        mixed5d = InceptionA(mixed5c, 288, 64, name='mixed5d')

        mixed6a = InceptionB(mixed5d, 288, name='mixed6a')

        mixed6b = InceptionC(mixed6a, 768, 128, name='mixed6b')
        mixed6c = InceptionC(mixed6b, 768, 160, name='mixed6c')
        mixed6d = InceptionC(mixed6c, 768, 160, name='mixed6d')
        mixed6e = InceptionC(mixed6d, 768, 192, name='mixed6e')

        mixed7a = InceptionD(mixed6e, 768, name='mixed7a')

        mixed7b = InceptionE(mixed7a, 1280, strategy, meshes, name='mixed7b')
        mixed7c = InceptionE(mixed7b, 2048, strategy, name='mixed7c')

        mean = mtf.reduce_mean(mixed7c, output_shape =
                mtf.Shape([mixed7c.shape[0], mixed7c.shape[-1]]))

        assert mean.shape[0].name == 'axis0' \
                and not mean.shape[1].name.startswith('axis')
        if strategy == 1:
            shape = mean.shape
            shape = shape.rename_dimension(shape[0].name,
                    mtf_labels.shape[0].name)
            shape = shape.rename_dimension(shape[1].name, 'axis0')
            with tf.variable_scope('reshape_mean'):
                mean = mtf.reshape(mean, shape)
            dim_name = 'axis1'
        elif strategy == 2:
            mean = mtf.rename_dimension(mean, 'axis0', mtf_labels.shape[0].name)
            dim_name = 'axis0'
        else:
            dim_name = RandName()
        fc = mtf.layers.dense(mean, mtf.Dimension(dim_name, num_classes))

        with tf.variable_scope('loss'):
            assert mtf_labels.mesh == fc.mesh
            assert mtf_labels.shape[0] == fc.shape[0]
            one_hot_labels = mtf.one_hot(mtf_labels, fc.shape[-1])
            cross_ent = mtf.layers.softmax_cross_entropy_with_logits(fc,
                    one_hot_labels, fc.shape[-1])
            loss = mtf.reduce_mean(cross_ent)

        with tf.variable_scope('optimize'):
            grads = mtf.gradients([loss], [v.outputs[0] for v in
                graph.trainable_variables])
            opt = mtf.optimize.SgdOptimizer(0.01)
            grad_updates = opt.apply_grads(grads, graph.trainable_variables)

        print(f'{datetime.datetime.now()} Beginning to lower mtf graph...',
                flush=True)
        lowering = mtf.Lowering(graph, mesh_to_impl)
        print(f'{datetime.datetime.now()} Finished lowering.', flush=True)
        tf_loss = lowering.export_to_tf_tensor(loss)
        tf_grad_updates = [lowering.lowered_operation(op) for op in grad_updates]

        # Initializer
        tf_init_vars = FlattenList([lowering.variables[var]
            .laid_out_tensor.all_slices for var in graph.all_variables])
        init_op = []
        for v in tf_init_vars:
            with tf.device(v.device):
                init_op.append(v.initializer)

        return init_op, tf_loss, tf_grad_updates


def main():
    parser = argparse.ArgumentParser(formatter_class =
            argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', type=int, required=False,
            default=128, help="Batch size.")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors.")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=5,
            help="No. of epochs")
    parser.add_argument('--display_steps', type=int, required=False, default=10,
            help="Steps to pass before displaying intermediate results")
    parser.add_argument('--max_steps', type=int, required=False, default=100,
            help='Maximum no. of steps to execute')
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(3)), 
            help="Strategy to use. 0: DataParallel, 1: Optimized, 2: Expert")
    parser.add_argument('--dataset_dir', type=str, required=False, default=None,
            help='Dataset directory')
    parser.add_argument('--labels_filename', type=str, required=False,
            default='labels.txt', help='Labels filename')
    parser.add_argument('--dataset_size', type=int, required=False,
            default=1000, help='Labels filename')
    args = parser.parse_args()
    [print(f'{arg} : {val}') for arg, val in vars(args).items()]

    if args.procs != 8:
        raise NotImplementedError('Current implementation only handles 8 GPUs.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(args.procs))[:-1]

    # Initalize the data generator
    dataset = ImageDataLoader(args.batch_size, (299, 299), dataset_size =
            args.dataset_size, dataset_dir=args.dataset_dir,
            labels_filename=args.labels_filename)
    train_batches_per_epoch = np.floor(dataset.dataset_size / args.batch_size).astype(np.int16)
    assert train_batches_per_epoch > 0
    
    # Input tensors
    tf_x, tf_y = dataset.next_batch()
    tf_x.set_shape([args.batch_size, 299, 299, 3])
    tf_y.set_shape([args.batch_size])

    init_ops, loss_op, grad_ops = Inception(tf_x, tf_y, args)

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
                    loss_val, *_ = sess.run([loss_op] + grad_ops)
                    step += 1
                    if step > args.max_steps:
                        break

                    if step % args.display_steps == 0 and step > 0:
                        print("Epoch: " + str(epoch) + "; Loss: " +
                                str(loss_val))

                dataset.reset_pointer()
            end = time.time()
            tot_time += (end - start)

    img_per_sec = (dataset.dataset_size * args.epochs) / tot_time
    print("Throughput: " + str(img_per_sec) + " images / sec")


if __name__ == '__main__':
    main()

