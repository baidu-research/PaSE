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
    global mesh_to_impl
    mesh_to_impl = {}

    if strategy == 0:
        mesh = mtf.Mesh(graph, 'mesh0')
        meshes.append(mesh)
        mesh_to_impl[mesh] = GetMeshImpl([8])

        mtf_img = mtf.import_tf_tensor(mesh, img, GetShape([('axis0', batch_size),
            h, w, ch]))
        mtf_labels = mtf.import_tf_tensor(mesh, labels, GetShape([('axis0',
            batch_size)]))

    elif strategy == 1:
        #device_lists = [[2], [2, 4], [4], [2, 2], [2, 2, 2]]

        # mesh0
        mesh = mtf.Mesh(graph, 'mesh0')
        meshes.append(mesh)
        devices = ['gpu:0', 'gpu:4']
        axes = ['axis1']
        mesh_to_impl[mesh] = GetMeshImpl([2], devices, axes)

        # mesh1
        mesh = mtf.Mesh(graph, 'mesh1')
        meshes.append(mesh)
        mesh_to_impl[mesh] = GetMeshImpl([4, 2])

        # mesh2
        mesh = mtf.Mesh(graph, 'mesh2')
        meshes.append(mesh)
        mesh_to_impl[mesh] = GetMeshImpl([4])

        # mesh3
        #mesh = mtf.Mesh(graph, 'mesh3')
        #meshes.append(mesh)
        #mesh_to_impl[mesh] = GetMeshImpl([2, 2])

        mtf_img = mtf.import_tf_tensor(meshes[0], img, GetShape([('axis1',
            batch_size), h, w, ch]))
        mtf_labels = mtf.import_tf_tensor(meshes[1], labels, GetShape([batch_size]))

    else:
        assert False

    return graph, meshes, mesh_to_impl, mtf_img, mtf_labels


def BasicConv(img, fltr, stride=(1,1), padding='VALID', dim_name=None,
        rename_dim = False, name=None):
    dim_name = RandName() if dim_name is None or dim_name in \
            img.shape.dimension_names else dim_name
    dim_names = img.shape.dimension_names[1:] + [dim_name]
    in_ch_dim_name = dim_names[-2]

    with tf.variable_scope(name, default_name='basic_conv'):
        conv = utils.Conv2d(img, GetFilterShape(fltr, dim_names), stride,
                padding)
        bn = mtf.layers.layer_norm(conv, conv.shape[0])
        if rename_dim:
            bn = mtf.rename_dimension(bn, bn.shape[-1].name, in_ch_dim_name)

        def LogDistribution(tsr, end='\n'):
            debug = True
            if debug:
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


def InceptionA(img, in_channels, pool_features, out_mesh_info=None,
        dim_name=None, name=None):
    with tf.variable_scope(name, default_name='inception_A'):
        ReplaceMesh = lambda x, cnt: utils.ReplaceMeshWithRemoval(out_mesh_info[0],
                x, out_mesh_info[1].shape[1], name='replace_mesh_%d' % cnt) \
                        if out_mesh_info is not None else x

        branch1x1 = BasicConv(img, ((1, 1, in_channels, 64)), dim_name=dim_name,
                name='branch1x1')
        branch1x1 = ReplaceMesh(branch1x1, 0)

        branch5x5 = BasicConv(img, ((1, 1, in_channels, 48)), dim_name=dim_name,
                name='branch5x5_1')
        branch5x5 = ReplaceMesh(branch5x5, 1)
        branch5x5 = BasicConv(branch5x5, ((5, 5, 48, 64)), padding='SAME',
                dim_name=dim_name, name='branch5x5_2')

        branch3x3dbl = BasicConv(img, ((1, 1, in_channels, 64)),
                dim_name=dim_name, name='branch3x3_1')
        branch3x3dbl = ReplaceMesh(branch3x3dbl, 2)
        branch3x3dbl = BasicConv(branch3x3dbl, ((3, 3, 64, 96)), padding='SAME',
                dim_name=dim_name, name='branch3x3_2')
        branch3x3dbl = BasicConv(branch3x3dbl, ((3, 3, 96, 96)), padding='SAME',
                dim_name=dim_name, name='branch3x3_3')

        branch_pool = utils.MaxPool(img, (3, 3), padding='SAME')
        branch_pool = BasicConv(branch_pool, ((1, 1, in_channels,
            pool_features)), dim_name=dim_name, name='branch_pool')
        branch_pool = ReplaceMesh(branch_pool, 3)

        return Concat([branch1x1, branch5x5, branch3x3dbl, branch_pool])


def InceptionB(img, in_channels, dim_name=None, name=None):
    with tf.variable_scope(name, default_name='inception_B'):
        branch3x3 = BasicConv(img, ((3, 3, in_channels, 384)), stride=2,
                dim_name=dim_name, rename_dim=True, name='branch3x3')

        branch3x3dbl = BasicConv(img, ((1, 1, in_channels, 64)),
                dim_name=dim_name, name='branch3x3_1')
        branch3x3dbl = BasicConv(branch3x3dbl, ((3, 3, 64, 96)), padding='SAME',
                dim_name=dim_name, rename_dim=True, name='branch3x3_2')
        branch3x3dbl = BasicConv(branch3x3dbl, ((3, 3, 96, 96)), stride=2,
                dim_name=dim_name, name='branch3x3_3')

        branch_pool = utils.MaxPool(img, (3, 3), stride=2)
        return Concat([branch3x3, branch3x3dbl, branch_pool])


def InceptionC(img, in_channels, channels_7x7, out_mesh_info=None,
        dim_name=None, rename_dim=False, name=None):
    with tf.variable_scope(name, default_name='inception_C'):
        branch1x1 = BasicConv(img, ((1, 1, in_channels, 192)),
                dim_name=dim_name, rename_dim=rename_dim, name='branch1x1')

        branch7x7 = BasicConv(img, ((1, 1, in_channels, channels_7x7)),
                dim_name=dim_name, rename_dim=rename_dim, name='branch7x7_1')
        branch7x7 = BasicConv(branch7x7, ((1, 7, channels_7x7, channels_7x7)),
                padding='SAME', dim_name=dim_name, rename_dim=rename_dim,
                name='branch7x7_2')
        branch7x7 = BasicConv(branch7x7, ((7, 1, channels_7x7, 192)),
                padding='SAME', dim_name=dim_name, rename_dim=rename_dim,
                name='branch7x7_3')

        branch7x7_dbl = BasicConv(img, ((1, 1, in_channels, channels_7x7)),
                dim_name=dim_name, rename_dim=rename_dim,
                name='branch7x7_dbl_1')
        branch7x7_dbl = BasicConv(branch7x7_dbl, ((7, 1, channels_7x7,
            channels_7x7)), padding='SAME', dim_name=dim_name,
            rename_dim=rename_dim, name='branch7x7_dbl_2')
        branch7x7_dbl = BasicConv(branch7x7_dbl, ((1, 7, channels_7x7,
            channels_7x7)), padding='SAME', dim_name=dim_name,
            rename_dim=rename_dim, name='branch7x7_dbl_3')
        branch7x7_dbl = BasicConv(branch7x7_dbl, ((7, 1, channels_7x7,
            channels_7x7)), padding='SAME', dim_name=dim_name,
            rename_dim=rename_dim, name='branch7x7_dbl_4')
        branch7x7_dbl = BasicConv(branch7x7_dbl, ((1, 7, channels_7x7, 192)),
                padding='SAME', dim_name=dim_name, rename_dim=rename_dim,
                name='branch7x7_dbl_5')

        branch_pool = utils.AvgPool(img, (3, 3), padding='SAME')
        branch_pool = BasicConv(branch_pool, ((1, 1, in_channels, 192)),
                dim_name=dim_name, rename_dim=rename_dim, name='branch_pool')
        return Concat([branch1x1, branch7x7, branch7x7_dbl, branch_pool])


def InceptionD(img, in_channels, dim_name=None, name=None):
    with tf.variable_scope(name, default_name='inception_D'):
        branch3x3 = BasicConv(img, ((1, 1, in_channels, 192)),
                dim_name=dim_name, name='branch3x3_1')
        branch3x3 = BasicConv(branch3x3, ((3, 3, 192, 320)), stride=2,
                dim_name=dim_name, name='branch3x3_2')

        branch7x7x3 = BasicConv(img, ((1, 1, in_channels, 192)),
                dim_name=dim_name, name='branch7x7x3_1')
        branch7x7x3 = BasicConv(branch7x7x3, ((1, 7, 192, 192)), padding='SAME',
                dim_name=dim_name, name='branch7x7x3_2')
        branch7x7x3 = BasicConv(branch7x7x3, ((7, 1, 192, 192)), padding='SAME',
                dim_name=dim_name, name='branch7x7x3_3')
        branch7x7x3 = BasicConv(branch7x7x3, ((3, 3, 192, 192)), stride=2,
                dim_name=dim_name, name='branch7x7x3_4')

        branch_pool = utils.MaxPool(img, (3, 3), stride=2)
        return Concat([branch3x3, branch7x7x3, branch_pool])


def InceptionE(img, in_channels, dim_name=None, rename_dim=False, name=None):
    with tf.variable_scope(name, default_name='inception_E'):
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

        branch_pool = utils.AvgPool(img, (3, 3), stride=1, padding='SAME')
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

        if strategy == 1:
            new_mesh = meshes[1]
            conv2a = utils.ReplaceMeshWithReplication(new_mesh, conv2a,
                    mesh_to_impl[new_mesh].shape[0],
                    name='replace_mesh0_with_mesh1')

        dim_name = 'axis0' if strategy == 1 else None
        conv2b = BasicConv(conv2a, (3, 3, 32, 64), padding='SAME',
                dim_name=dim_name, name='conv2b')
        pool = utils.MaxPool(conv2b, (3, 3), stride=2, name='pool1')

        if strategy == 1:
            curr_dims = pool.shape.dims
            assert curr_dims[0].name == 'axis1' and curr_dims[-1].name == 'axis0'
            new_dims = curr_dims[:]
            new_dims[0] = new_dims[0]._replace(name=curr_dims[-1].name)
            new_dims[-1] = new_dims[-1]._replace(name=RandName())
            with tf.variable_scope('reshape_pool1'):
                pool = mtf.reshape(pool, new_dims)

            new_mesh = meshes[2]
            pool = utils.ReplaceMeshWithRemoval(new_mesh, pool,
                    mesh_to_impl[pool.mesh].shape[1],
                    name='replace_pool_mesh1_with_mesh2')

        conv3b = BasicConv(pool, (1, 1, 64, 80), name='conv3b')

        if strategy == 1:
            new_mesh = meshes[1]
            conv3b = utils.ReplaceMeshWithReplication(new_mesh, conv3b,
                    mesh_to_impl[new_mesh].shape[1],
                    name='replace_conv3b_mesh2_with_mesh1')

        dim_name = 'axis1' if strategy == 1 else None
        conv4a = BasicConv(conv3b, (3, 3, 80, 192), dim_name=dim_name,
                name='conv4a')
        pool = utils.MaxPool(conv4a, (3, 3), stride=2, name='pool2')

        out_mesh_info = (meshes[2], mesh_to_impl[meshes[1]]) \
                if strategy == 1 else None
        mixed5b = InceptionA(pool, 192, 32, out_mesh_info, name='mixed5b')
        mixed5c = InceptionA(mixed5b, 256, 64, name='mixed5c')
        mixed5d = InceptionA(mixed5c, 288, 64, name='mixed5d')

        if strategy == 1:
            new_mesh = meshes[1]
            mixed5d = utils.ReplaceMeshWithReplication(new_mesh, mixed5d,
                    mesh_to_impl[new_mesh].shape[1],
                    name='replace_mixed5d_mesh2_with_mesh1')

        dim_name = 'axis1' if strategy == 1 else None
        mixed6a = InceptionB(mixed5d, 288, dim_name=dim_name, name='mixed6a')

        if strategy == 1:
            new_mesh = meshes[2]
            mixed6a = utils.ReplaceMeshWithRemoval(new_mesh, mixed6a,
                    mesh_to_impl[mixed6a.mesh].shape[1],
                    name='replace_mixed6a_mesh1_with_mesh2')

        mixed6b = InceptionC(mixed6a, 768, 128, name='mixed6b')

        if strategy == 1:
            new_mesh = meshes[1]
            mixed6b = utils.ReplaceMeshWithReplication(new_mesh, mixed6b,
                    mesh_to_impl[new_mesh].shape[1],
                    name='replace_mixed6b_mesh2_with_mesh1')

        dim_name = 'axis1' if strategy == 1 else None
        mixed6c = InceptionC(mixed6b, 768, 160, dim_name=dim_name,
                rename_dim=True, name='mixed6c')
        mixed6d = InceptionC(mixed6c, 768, 160, dim_name=dim_name,
                rename_dim=True, name='mixed6d')
        mixed6e = InceptionC(mixed6d, 768, 192, dim_name=dim_name,
                rename_dim=True, name='mixed6e')

        dim_name = 'axis1' if strategy == 1 else None
        mixed7a = InceptionD(mixed6e, 768, dim_name=dim_name, name='mixed7a')

        mixed7b = InceptionE(mixed7a, 1280, dim_name=dim_name,
                rename_dim=True, name='mixed7b')
        mixed7c = InceptionE(mixed7b, 2048, dim_name=dim_name,
                rename_dim=True, name='mixed7c')

        mean = mtf.reduce_mean(mixed7c, output_shape =
                mtf.Shape([mixed7c.shape[0], mixed7c.shape[-1]]))

        if strategy == 1:
            new_shape = mean.shape.dims
            new_shape[0] = mtf_labels.shape[0]
            new_shape[1] = new_shape[1]._replace(name='axis1')
            new_shape = mtf.Shape(new_shape)
            with tf.variable_scope('reshape_mean'):
                mean = mtf.reshape(mean, new_shape)
        
        dim_name = 'axis0' if strategy == 1 else RandName()
        fc = mtf.layers.dense(mean, mtf.Dimension(dim_name, num_classes))

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

        print('Beginning to lower mtf graph...', flush=True)
        lowering = mtf.Lowering(graph, mesh_to_impl)
        print('Finished lowering.', flush=True)
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

        return init_op, tf_loss, tf_grad_updates


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
    
    for arg, val in vars(args).items():
        print(str(arg) + ": " + str(val))
    print()
            
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

