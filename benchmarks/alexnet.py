import numpy as np
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf

import sys
import time
import os, os.path
import string, random

import common
from dataloader import ImageDataLoader
import utils
from mesh_transformations import ReplaceMeshWithDuplicates, ReplaceMeshWithIndependentAxes
from mtf_operations import Conv2d, MaxPool
#import dgx_mesh_impl


def GetShape(dims):
    sh = []
    for d in dims:
        try:
            name, size = d
        except (TypeError, ValueError):
            name, size = utils.RandName(), d
        sh.append(mtf.Dimension(name, size))

    sh = mtf.Shape(sh)
    return sh

def CreateMeshes(img, labels, num_nodes, num_gpus, args):
    h, w, ch = 227, 227, 3
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}

    strategy = args.strategy
    batch_size = args.batch_size

    def GetMeshImpl(dev_cnts, devices=None, node_cnt=num_nodes):
        return utils.GetMeshImpl(dev_cnts, devices=devices, num_nodes=node_cnt)
                #mesh_impl=dgx_mesh_impl.DGXMeshImpl)

    if strategy == 0:
        mesh = mtf.Mesh(graph, 'mesh0')
        meshes.append(mesh)
        mesh_to_impl[mesh] = GetMeshImpl([num_gpus])

        mtf_img = mtf.import_tf_tensor(mesh, img, GetShape([('axis0', batch_size),
            h, w, ch]))
        mtf_labels = mtf.import_tf_tensor(mesh, labels, GetShape([('axis0',
            batch_size)]))

    elif strategy == 1:
        mesh = mtf.Mesh(graph, 'mesh0')
        meshes.append(mesh)
        mesh_to_impl[mesh] = GetMeshImpl([num_gpus])

        if num_gpus == 8:
            mesh = mtf.Mesh(graph, 'mesh1')
            meshes.append(mesh)
            mesh_to_impl[mesh] = GetMeshImpl([4, 2]) 

            mtf_img = mtf.import_tf_tensor(meshes[0], img, GetShape([('axis0',
                batch_size), h, w, ch]))
            mtf_labels = mtf.import_tf_tensor(meshes[1], labels,
                    GetShape([batch_size]))

        elif num_gpus == 16:
            mesh = mtf.Mesh(graph, 'mesh1')
            meshes.append(mesh)
            mesh_to_impl[mesh] = mesh_impl1 = GetMeshImpl([2, 8])

            mesh = mtf.Mesh(graph, 'mesh2')
            meshes.append(mesh)
            devices = [mesh_impl1.devices[0], mesh_impl1.devices[8]]
            mesh_to_impl[mesh] = GetMeshImpl([2, 1], devices=devices)

            mtf_img = mtf.import_tf_tensor(meshes[0], img, GetShape([('axis0',
                batch_size), h, w, ch]))
            mtf_labels = mtf.import_tf_tensor(meshes[2], labels,
                    GetShape([batch_size]))

        elif num_gpus == 32:
            mesh = mtf.Mesh(graph, 'mesh1')
            meshes.append(mesh)
            mesh_to_impl[mesh] = mesh_impl1 = GetMeshImpl([4, 8])

            mesh = mtf.Mesh(graph, 'mesh2')
            meshes.append(mesh)
            devices = [mesh_impl1.devices[i] for i in range(0, 32, 8)]
            mesh_to_impl[mesh] = GetMeshImpl([4, 1], devices=devices)

            mtf_img = mtf.import_tf_tensor(meshes[0], img, GetShape([('axis0',
                batch_size), h, w, ch]))
            mtf_labels = mtf.import_tf_tensor(meshes[2], labels,
                    GetShape([batch_size]))

        else:
            assert False

    elif strategy == 2:
        mesh = mtf.Mesh(graph, 'mesh0')
        meshes.append(mesh)
        mesh_to_impl[mesh] = GetMeshImpl([num_gpus])

        mtf_img = mtf.import_tf_tensor(mesh, img, GetShape([('axis0', batch_size),
            h, w, ch]))
        mtf_labels = mtf.import_tf_tensor(mesh, labels, GetShape([batch_size]))

    elif strategy == 3:
        mesh = mtf.Mesh(graph, 'mesh0')
        meshes.append(mesh)
        mesh_to_impl[mesh] = GetMeshImpl([num_gpus])

        mesh = mtf.Mesh(graph, 'mesh1')
        meshes.append(mesh)
        mesh_to_impl[mesh] = GetMeshImpl([1], node_cnt=1)

        mtf_img = mtf.import_tf_tensor(meshes[0], img, GetShape([('axis0',
            batch_size), h, w, ch]))
        mtf_labels = mtf.import_tf_tensor(meshes[1], labels,
                GetShape([batch_size]))

    else:
        assert False

    return graph, meshes, mesh_to_impl, mtf_img, mtf_labels

def GetFilterShape(img, sizes):
    names = img.shape.dimension_names[1:]
    return mtf.Shape([mtf.Dimension(names[0], sizes[0]),
        mtf.Dimension(names[1], sizes[1]),
        mtf.Dimension(names[2], sizes[2]),
        mtf.Dimension(utils.RandName(), sizes[3])])

def Alexnet(img, labels, num_nodes, num_gpus, args):
    num_classes = 1000
    keep_prob = 0.5
    learning_rate = 0.01
    graph, meshes, mesh_to_impl, mtf_img, mtf_labels = CreateMeshes(img, labels,
            num_nodes, num_gpus, args)
    RenameFC = lambda x: mtf.rename_dimension(x, x.shape[-1].name, utils.RandName())

    strategy = args.strategy
    if strategy == 0:
        fc6_units = mtf.Dimension(utils.RandName(), 4096)
        fc7_units = mtf.Dimension(utils.RandName(), 4096)
        fc8_units = mtf.Dimension(utils.RandName(), num_classes)

    elif strategy == 1:
        if num_nodes == 1:
            fc6_units = mtf.Dimension('axis1', 4096)
            fc7_units = mtf.Dimension('axis0', 4096)
            fc8_units = mtf.Dimension('axis1', num_classes)
        elif num_nodes >= 2:
            fc6_units = mtf.Dimension('axis0', 4096)
            fc7_units = mtf.Dimension('axis1', 4096)
            fc8_units = mtf.Dimension('axis0', num_classes)
        else:
            assert False

    elif strategy == 2:
        num_classes = num_classes + num_gpus - (num_classes % num_gpus)
        fc6_units = mtf.Dimension('axis0', 4096)
        fc7_units = mtf.Dimension('axis0', 4096)
        fc8_units = mtf.Dimension('axis0', num_classes)

    elif strategy == 3:
        fc6_units = mtf.Dimension(utils.RandName(), 4096)
        fc7_units = mtf.Dimension(utils.RandName(), 4096)
        fc8_units = mtf.Dimension(utils.RandName(), num_classes)

    with tf.variable_scope('alexnet'):
        # Conv1 + ReLU + maxpool1
        conv1 = Conv2d(mtf_img, GetFilterShape(mtf_img, (11, 11, 3, 96)),
                (4, 4), 'VALID', activation=mtf.relu, name='conv1')
        pool1 = MaxPool(conv1, (3, 3), (2, 2), 'VALID', name='pool1')

        # Conv2 + ReLU + maxpool2
        conv2 = Conv2d(pool1, GetFilterShape(pool1, (5, 5, 96, 256)), (1,
            1), 'SAME', activation=mtf.relu, name='conv2')
        pool2 = MaxPool(conv2, (3, 3), (2, 2), name='pool2')

        # Conv3 + ReLU
        conv3 = Conv2d(pool2, GetFilterShape(pool2, (3, 3, 256, 384)),
                padding='SAME', activation=mtf.relu, name='conv3')

        # Conv4 + ReLU
        conv4 = Conv2d(conv3, GetFilterShape(conv3, (3, 3, 384, 384)),
                padding='SAME', activation=mtf.relu, name='conv4')

        # Conv5 + ReLU + maxpool5
        conv5 = Conv2d(conv4, GetFilterShape(conv4, (3, 3, 384, 256)),
                padding='SAME', activation=mtf.relu, name='conv5')
        pool5 = MaxPool(conv5, (3, 3), (2, 2), name='pool5')

        # Rename dims
        if strategy == 1:
            k_dim = mtf.Dimension(utils.RandName(),
                    utils.Prod(pool5.shape.to_integer_list[1:]))
            pool5 = mtf.reshape(pool5, mtf.Shape([pool5.shape[0], k_dim]))
            if num_nodes == 1:
                pool5 = ReplaceMeshWithIndependentAxes(pool5, meshes[1],
                        (utils.RandName(), 'axis0'))
            elif num_nodes >= 2:
                pool5 = ReplaceMeshWithIndependentAxes(pool5, meshes[1],
                        (utils.RandName(), 'axis1'))
            else:
                assert False

        elif strategy == 2:
            pool5 = mtf.rename_dimension(pool5, pool5.shape[0].name, utils.RandName())

        elif strategy == 3:
            assert pool5.shape[0].name == 'axis0'
            dim_names = pool5.shape.rename_dimension('axis0', utils.RandName())
            pool5 = ReplaceMeshWithIndependentAxes(pool5, meshes[1], dim_names)

        # FC + ReLU + dropout
        fc_activation = lambda x: mtf.dropout(mtf.relu(x), keep_prob)
        fc6 = mtf.layers.dense(pool5, fc6_units, activation=fc_activation,
                reduced_dims=pool5.shape[1:], name='fc6')
        if strategy == 2:
            fc6 = RenameFC(fc6)

        fc7 = mtf.layers.dense(fc6, fc7_units, activation=fc_activation,
                reduced_dims=fc6.shape.dims[-1:], name='fc7')
        if strategy == 2:
            fc7 = RenameFC(fc7)

        fc8 = mtf.layers.dense(fc7, fc8_units, reduced_dims=fc7.shape.dims[-1:], name='fc8')
        fc8 = mtf.dropout(fc8, keep_prob)

        if strategy == 1 and num_nodes >= 2:
            assert fc8.shape[-1].name == 'axis0'
            fc8 = ReplaceMeshWithDuplicates(fc8, meshes[2])

    with tf.variable_scope('loss'):
        if fc8.shape[0] != mtf_labels.shape[0]:
            fc8 = mtf.rename_dimension(fc8, fc8.shape[0].name,
                    mtf_labels.shape[0].name)
        one_hot_labels = mtf.one_hot(mtf_labels, fc8.shape[-1])
        mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(fc8,
                one_hot_labels, fc8.shape[-1])
        mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    with tf.variable_scope('optimize'):
        grads = mtf.gradients([mtf_loss], [v.outputs[0] for v in
            graph.trainable_variables])
        opt = mtf.optimize.SgdOptimizer(learning_rate)
        grad_updates = opt.apply_grads(grads, graph.trainable_variables)

    print('Beginning to lower mtf graph...', flush=True)
    lowering = mtf.Lowering(graph, mesh_to_impl)
    print('Finished lowering.', flush=True)
    tf_loss = lowering.export_to_tf_tensor(mtf_loss)
    tf_grad_updates = [lowering.lowered_operation(op) for op in grad_updates]

    # Initializer
    tf_init_vars = \
            utils.FlattenList([lowering.variables[var].laid_out_tensor.all_slices
                for var in graph.trainable_variables])
    init_op = []
    for v in tf_init_vars:
        with tf.device(v.device):
            init_op.append(v.initializer)

    return init_op, tf_loss, tf_grad_updates


def main():
    t = common.Trainer()

    # Initalize the data generator
    args = t.args
    dataset = ImageDataLoader(args.batch_size, (227, 227), dataset_size =
            args.dataset_size, dataset_dir=args.dataset_dir,
            labels_filename=args.labels_filename)
    
    # Input tensors
    tf_x, tf_y = dataset.next_batch()
    tf_x.set_shape([args.batch_size, 227, 227, 3])
    tf_y.set_shape([args.batch_size])

    model = Alexnet(tf_x, tf_y, t.num_nodes, t.num_gpus, args)
    t.train(*model, dataset)


if __name__ == '__main__':
    main()

