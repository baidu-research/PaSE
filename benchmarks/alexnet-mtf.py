import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import sys
import time
import os
from argparse import ArgumentParser

from dataloader import ImageDataLoader
import utils


class ImportToMeshBackpropOperation(mtf.Operation):
    def __init__(self, mesh, input, name=None):
        super().__init__([input], mesh=mesh, name=name or
                'import_to_mesh_backprop')
        self._outputs = [mtf.Tensor(self, input.shape, input.dtype)]

    def lower(self, lowering):
        input_slices = \
                lowering.tensors[self.inputs[0]].to_laid_out_tensor().tensor_list
        n_slices = len(input_slices)
        assert n_slices % 2 == 0
        output_slices = [t for t in input_slices[:int(n_slices/2)]]

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


class ImportToMeshOperation(mtf.Operation):
    def __init__(self, mesh, input, name=None):
        super().__init__([input], mesh=mesh, name=name or 'import_to_mesh')
        self.old_mesh = input.mesh
        self._outputs = [mtf.Tensor(self, input.shape, input.dtype)]

    def gradient(self, grad_ys):
        dy = grad_ys[0]
        return ImportToMeshBackpropOperation(self.old_mesh, dy).outputs

    def lower(self, lowering):
        input_slices = \
                lowering.tensors[self.inputs[0]].to_laid_out_tensor().tensor_list
        output_slices = []
        for t in input_slices:
            output_slices.append([t, t])
        output_slices = utils.FlattenList(utils.TransposeLists(output_slices))

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def import_to_mesh(mesh, tsr, name):
    return ImportToMeshOperation(mesh, tsr, name).outputs[0]


def main():
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, required=False, default=256,
            help="Batch size. (Default: 256)")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors. (Default: 8)")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
            help="No. of epochs")
    parser.add_argument('--display_steps', type=int, required=False, default=10,
            help="No. of epochs")
    parser.add_argument('-d', '--dropout', type=float, required=False,
            default=0.5, help="keep_prob value for dropout layers. (Default: 0.5)")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(3)), 
            help="Strategy to use. 0: DataParallel, \
                    1: Optimized. \
                    (Default: 0) ")
    parser.add_argument('--dataset_dir', type=str, required=False, default=None,
            help='Dataset directory')
    parser.add_argument('--labels_filename', type=str, required=False,
            default='labels.txt', help='Labels filename')
    parser.add_argument('--dataset_size', type=int, required=False,
            default=1000, help='Labels filename')
    args = vars(parser.parse_args())

    # Input parameters
    num_gpus = args['procs']
    batch_size = args['batch']
    keep_prob = args['dropout']
    num_epochs = args['epochs']
    strategy = args['strategy']
    num_classes = 1000
    learning_rate = 0.01
    display_step = args['display_steps']
    warmup = 10

    if num_gpus != 8 and strategy > 0:
        raise NotImplementedError('Current implementation only handles 8 GPUs \
                for model parallel strategies.')

    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(num_gpus))[:-1]
    
    # Initalize the data generator seperately for the training and validation set
    dataset_dir = args['dataset_dir']
    labels_filename = args['labels_filename']
    dataset_size = args['dataset_size']
    dataset = ImageDataLoader(batch_size, (227, 227), dataset_size=dataset_size,
            dataset_dir=dataset_dir, labels_filename=labels_filename)
    train_batches_per_epoch = np.floor(dataset.dataset_size / batch_size).astype(np.int16)
    assert train_batches_per_epoch > 0
    
    # Input tensors
    tf_x, tf_y = dataset.next_batch()
    tf_x.set_shape([batch_size, 227, 227, 3])
    tf_y.set_shape([batch_size])

    # mtf dimensions
    batch_dim = mtf.Dimension('batch_dim', batch_size)
    in_ch_dim = mtf.Dimension('in_ch_dim', 3)
    h_dim = mtf.Dimension('h_dim', 227)
    w_dim = mtf.Dimension('w_dim', 227)
    fc_batch_dim = mtf.Dimension('fc_batch_dim', batch_size)

    def AssignLayout(ta_axes, mesh_axis):
        layout = []
        for a in ta_axes:
            layout.append((a, mesh_axis))
        return layout

    # mtf graph
    graph = mtf.Graph()

    # mtf mesh layouts for different strategies
    if strategy == 0:
        mesh = mtf.Mesh(graph, 'mesh')
        mesh_shape = [('p1', num_gpus)]
        devices = ['gpu:%d' %i for i in range(num_gpus)]
        layout = AssignLayout([batch_dim.name, fc_batch_dim.name], 'p1')
        mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape,
                layout, devices)

        meshes = {mesh:mesh_impl}

        mtf_x = mtf.import_tf_tensor(mesh, tf_x, mtf.Shape([batch_dim, h_dim,
            w_dim, in_ch_dim]))
        mtf_y = mtf.import_tf_tensor(mesh, tf_y, mtf.Shape([fc_batch_dim]))
    elif strategy == 1:
        # mtf 1D mesh
        mesh1 = mtf.Mesh(graph, 'mesh1')
        mesh_shape = [('m1_p1', 4)]
        devices = ['gpu:%d' % i for i in range(4)]
        layout = AssignLayout([batch_dim.name], 'm1_p1')
        mesh1_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape,
                layout, devices)

        # mtf 2D mesh
        mesh2 = mtf.Mesh(graph, 'mesh2')
        mesh_shape = [('m2_p1', 4), ('m2_p2', 2)]
        devices = ['gpu:%d' % i for i in range(8)]
        p1_layout = AssignLayout([batch_dim.name, 'fc_n_dim'], 'm2_p1')
        p2_layout = AssignLayout(['n_dim', 'fc_k_dim'], 'm2_p2')
        layout = p1_layout + p2_layout
        mesh2_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout,
                devices)

        meshes = {mesh1:mesh1_impl, mesh2:mesh2_impl}

        # mtf input / output variables
        mtf_x = mtf.import_tf_tensor(mesh1, tf_x, mtf.Shape([batch_dim, h_dim,
            w_dim, in_ch_dim]))
        mtf_y = mtf.import_tf_tensor(mesh2, tf_y, mtf.Shape([fc_batch_dim]))
    else:
        assert False
        '''
        mesh = mtf.Mesh(graph, 'mesh')
        mesh_shape = [('p1', 4), ('p2', 2)]
        devices = ['gpu:%d' %i for i in range(num_gpus)]
        p1_layout = AssignLayout([batch_dim.name, 'fc_n_dim'], 'p1')
        p2_layout = AssignLayout(['n_dim', 'fc_k_dim'], 'p2')
        layout = p1_layout + p2_layout
        mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape,
                layout, devices)

        meshes = {mesh:mesh_impl}

        mtf_x = mtf.import_tf_tensor(mesh, tf_x, mtf.Shape([batch_dim, h_dim,
            w_dim, in_ch_dim]))
        mtf_y = mtf.import_tf_tensor(mesh, tf_y, mtf.Shape([fc_batch_dim]))
        '''

    with tf.variable_scope('alexnet'):
        ConvRename = lambda x: mtf.rename_dimension(x, x.shape[-1].name,
                in_ch_dim.name)
        FCRename = lambda x: mtf.rename_dimension(x, x.shape[-1].name,
                'fc_k_dim')
        FltrShape = lambda f: (mtf.Dimension('r_dim', f[0]),
                mtf.Dimension('s_dim', f[1]), mtf.Dimension('in_ch_dim', f[2]),
                mtf.Dimension('n_dim', f[3]))

        # Conv1 + ReLU + maxpool1
        conv1 = utils.Conv2d(mtf_x, FltrShape((11, 11, 3, 96)), (4, 4), 'VALID',
                activation=mtf.relu, name='conv1')
        pool1 = utils.MaxPool(conv1, (3, 3), (2, 2), 'VALID', name='pool1')
        pool1 = ConvRename(pool1)

        # Import pool1 to mesh2
        if strategy == 1:
            pool1 = import_to_mesh(mesh2, pool1, name='import_to_mesh2')

        # Conv2 + ReLU + maxpool2
        conv2 = utils.Conv2d(pool1, FltrShape((5, 5, 96, 256)), (1, 1), 'SAME',
                activation=mtf.relu, name='conv2')
        pool2 = utils.MaxPool(conv2, (3, 3), (2, 2), name='pool2')
        pool2 = ConvRename(pool2)

        # Conv3 + ReLU
        conv3 = utils.Conv2d(pool2, FltrShape((3, 3, 256, 384)), padding='SAME',
                activation=mtf.relu, name='conv3')
        conv3 = ConvRename(conv3)

        # Conv4 + ReLU
        conv4 = utils.Conv2d(conv3, FltrShape((3, 3, 384, 384)), padding='SAME',
                activation=mtf.relu, name='conv4')
        conv4 = ConvRename(conv4)

        # Conv5 + ReLU + maxpool5
        conv5 = utils.Conv2d(conv4, FltrShape((3, 3, 384, 256)), padding='SAME',
                activation=mtf.relu, name='conv4')
        pool5 = utils.MaxPool(conv5, (3, 3), (2, 2), name='pool5')

        # Rename dims
        pool5 = mtf.rename_dimension(pool5, pool5.shape[0].name,
                fc_batch_dim.name)

        # FC + ReLU + dropout
        fc_activation = lambda x: mtf.dropout(mtf.relu(x), keep_prob)
        fc6 = mtf.layers.dense(pool5, mtf.Dimension('fc_n_dim', 4096),
                activation=fc_activation, reduced_dims=pool5.shape[1:],
                name='fc6')
        fc6 = FCRename(fc6)
        fc7 = mtf.layers.dense(fc6, mtf.Dimension('fc_n_dim', 4096),
                activation=fc_activation, name='fc7')
        fc7 = FCRename(fc7)
        fc8 = mtf.layers.dense(fc7, mtf.Dimension('fc_n_dim', num_classes),
                name='fc8')
        fc8 = mtf.dropout(fc8, keep_prob)

    with tf.variable_scope('loss'):
        one_hot_labels = mtf.one_hot(mtf_y, fc8.shape[-1])
        mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(fc8,
                one_hot_labels, fc8.shape[-1])
        mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    with tf.variable_scope('optimize'):
        grads = mtf.gradients([mtf_loss], [v.outputs[0] for v in
            graph.trainable_variables])
        opt = mtf.optimize.SgdOptimizer(learning_rate)
        grad_updates = opt.apply_grads(grads, graph.trainable_variables)

    lowering = mtf.Lowering(graph, meshes)
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

    # Training
    with tf.name_scope('train'):
        with tf.Session() as sess:
            dataset.reset_pointer()
            sess.run(init_op)

            tot_time = float(0)
            start = time.time()
            for epoch in range(num_epochs):
                for step in range(train_batches_per_epoch):
                    loss_val, *rem = sess.run([tf_loss] + tf_grad_updates)

                if epoch % display_step == 0 and epoch > 0:
                    print("Epoch: " + str(epoch) + "; Loss: " + str(loss_val))

                dataset.reset_pointer()
            end = time.time()
            tot_time += (end - start)

    img_per_sec = (dataset.dataset_size * num_epochs) / tot_time
    print("Throughout: " + str(img_per_sec) + " images / sec")


if __name__ == '__main__':
    main()

