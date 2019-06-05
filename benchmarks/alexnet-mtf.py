import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import sys
import time
import os
from datetime import datetime
from argparse import ArgumentParser
import functools
import random
import string

from dataloader import ImageDataLoader


def FlattenList(l):
   return [item for sublist in l for item in sublist]


def TransposeLists(l):
    return [list(x) for x in zip(*l)]


class Conv2dOperation(mtf.Conv2dOperation):
    def __init__(self, conv_input, conv_filter, strides, padding, name=None):
        mtf.Operation.__init__(self, [conv_input, conv_filter], name=name or
                "conv2d")
        self._padding = padding
        self._batch_dims = conv_input.shape.dims[:-3]
        self._in_h_dim, self._in_w_dim, self._in_dim = conv_input.shape.dims[-3:]
        self._fh_dim, self._fw_dim = conv_filter.shape.dims[:2]
        f_in_dim, self._out_dim = conv_filter.shape.dims[2:]
        if f_in_dim != self._in_dim:
          raise ValueError("Dimensions do not match input=%s filter=%s"
                           % (conv_input, conv_filter))
        out_h = self._in_h_dim.size
        out_w = self._in_w_dim.size
        if padding == "VALID":
            out_h -= self._fh_dim.size
            out_w -= self._fw_dim.size

        self._strides = strides
        if strides is not None:
            out_h //= strides[1]
            out_w //= strides[2]

        if padding == "VALID":
            out_h += 1
            out_w += 1

        self._out_h_dim = mtf.Dimension(self._in_h_dim.name, out_h)
        self._out_w_dim = mtf.Dimension(self._in_w_dim.name, out_w)
        output_shape = mtf.Shape(
            self._batch_dims + [self._out_h_dim, self._out_w_dim, self._out_dim])
        self._outputs = [mtf.Tensor(self, output_shape, conv_input.dtype)]


def Conv2d(tsr, fltr_shape, stride=(1,1), padding='VALID', use_bias=True,
        activation=None, name=None):
    with tf.variable_scope(name, default_name='conv2d'):
        assert tsr.shape[-1] == fltr_shape[-2]

        w = mtf.get_variable(tsr.mesh, 'weight', fltr_shape, dtype=tsr.dtype)
        out = Conv2dOperation(tsr, w, (1, stride[0], stride[1], 1),
                padding).outputs[0]

        if use_bias == True:
            b = mtf.get_variable(tsr.mesh, 'bias', mtf.Shape([out.shape[-1]]),
                    initializer=tf.zeros_initializer(), dtype=tsr.dtype)
            out += b

        if activation is not None:
            out = activation(out)

        return out


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
        n_slices_by_2 = int(n_slices / 2)

        output_slices = []
        for t1, t2 in zip(input_slices, input_slices[n_slices_by_2:]):
            with tf.device(t1.device):
                output_slices.append(t1 + t2)

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
        output_slices = FlattenList(TransposeLists(output_slices))

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def import_to_mesh(mesh, tsr, name):
    return ImportToMeshOperation(mesh, tsr, name).outputs[0]


def MaxPool(tsr, fltr, stride=(1,1), padding='VALID', name=None):
    with tf.variable_scope(name, default_name='pool'):
        def max_pool(x):
            return tf.nn.max_pool(x, [1, fltr[0], fltr[1], 1], [1, stride[0],
                stride[1], 1], padding)

        # Output shape
        h_o = tsr.shape[1].size
        w_o = tsr.shape[2].size
        if padding == 'VALID':
            h_o -= fltr[0]
            w_o -= fltr[1]
        h_o //= stride[0]
        w_o //= stride[1]
        if padding == 'VALID':
            h_o += 1
            w_o += 1

        output_shape = tsr.shape.resize_dimension(tsr.shape[1].name, h_o)
        output_shape = output_shape.resize_dimension(tsr.shape[2].name, w_o)

        splittable_dims = [tsr.shape.dims[0], tsr.shape.dims[-1]]
        out = mtf.slicewise(max_pool, [tsr], output_shape, tsr.dtype, splittable_dims)
        return out


def main():
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, required=False, default=256,
            help="Batch size. (Default: 256)")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors. (Default: 8)")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
            help="No. of epochs")
    parser.add_argument('-d', '--dropout', type=float, required=False,
            default=0.5, help="keep_prob value for dropout layers. (Default: 0.5)")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(2)), 
            help="Strategy to use. 0: DataParallel, 1: Optimized. (Default: 0)")
    parser.add_argument('dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('labels_filename', type=str, help='Labels filename')
    args = vars(parser.parse_args())

    # Input parameters
    num_gpus = args['procs']
    batch_size = args['batch']
    keep_prob = args['dropout']
    num_epochs = args['epochs']
    strategy = args['strategy']
    num_classes = 1000
    learning_rate = 0.01
    display_step = 10
    warmup = 10

    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(num_gpus))[:-1]
    
    # Initalize the data generator seperately for the training and validation set
    dataset_dir = args['dataset_dir']
    labels_filename = args['labels_filename']
    dataset = ImageDataLoader(batch_size, dataset_dir, labels_filename, 32, 8)
    train_batches_per_epoch = np.floor(dataset.data_size / batch_size).astype(np.int16)
    
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

    # mtf 1D mesh
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_shape_1d = [('m1_p1', 4),]
    devices = ['gpu:%d' % i for i in range(4)]
    layout = AssignLayout([batch_dim.name], 'm1_p1')
    mesh1_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape_1d,
            layout, devices)

    # mtf 2D mesh
    mesh2 = mtf.Mesh(graph, 'mesh2')
    mesh_shape_2d = [('m2_p1', 4), ('m2_p2', 2)]
    devices = ['gpu:%d' % i for i in range(8)]
    p1_layout = AssignLayout([batch_dim.name, 'fc_n_dim'], 'm2_p1')
    p2_layout = AssignLayout(['n_dim', 'fc_k_dim'], 'm2_p2')
    layout = p1_layout + p2_layout
    mesh2_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape_2d, layout,
            devices)

    meshes = {mesh1:mesh1_impl, mesh2:mesh2_impl}

    # mtf input / output variables
    mtf_x = mtf.import_tf_tensor(mesh1, tf_x, mtf.Shape([batch_dim, h_dim,
        w_dim, in_ch_dim]))
    mtf_y = mtf.import_tf_tensor(mesh2, tf_y, mtf.Shape([fc_batch_dim]))

    with tf.variable_scope('alexnet'):
        ConvRename = lambda x: mtf.rename_dimension(x, x.shape[-1].name,
                in_ch_dim.name)
        FCRename = lambda x: mtf.rename_dimension(x, x.shape[-1].name,
                'fc_k_dim')
        FltrShape = lambda f: (mtf.Dimension('r_dim', f[0]),
                mtf.Dimension('s_dim', f[1]), mtf.Dimension('in_ch_dim', f[2]),
                mtf.Dimension('n_dim', f[3]))

        # Conv1 + ReLU + maxpool1
        conv1 = Conv2d(mtf_x, FltrShape((11, 11, 3, 96)), (4, 4), 'VALID',
                activation=mtf.relu, name='conv1')
        pool1_mesh1 = MaxPool(conv1, (3, 3), (2, 2), 'VALID', name='pool1')
        pool1_mesh1 = ConvRename(pool1_mesh1)

        # Import pool1 to mesh2
        pool1_mesh2 = import_to_mesh(mesh2, pool1_mesh1, name='import_to_mesh2')

        # Conv2 + ReLU + maxpool2
        conv2 = Conv2d(pool1_mesh2, FltrShape((5, 5, 96, 256)), (1, 1), 'SAME',
                activation=mtf.relu, name='conv2')
        pool2 = MaxPool(conv2, (3, 3), (2, 2), name='pool2')
        pool2 = ConvRename(pool2)

        # Conv3 + ReLU
        conv3 = Conv2d(pool2, FltrShape((3, 3, 256, 384)), padding='SAME',
                activation=mtf.relu, name='conv3')
        conv3 = ConvRename(conv3)

        # Conv4 + ReLU
        conv4 = Conv2d(conv3, FltrShape((3, 3, 384, 384)), padding='SAME',
                activation=mtf.relu, name='conv4')
        conv4 = ConvRename(conv4)

        # Conv5 + ReLU + maxpool5
        conv5 = Conv2d(conv4, FltrShape((3, 3, 384, 256)), padding='SAME',
                activation=mtf.relu, name='conv4')
        pool5 = MaxPool(conv5, (3, 3), (2, 2), name='pool5')

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
            FlattenList([lowering.variables[var].laid_out_tensor.all_slices for
                var in graph.trainable_variables])
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

                    if step % display_step == 0 and step > 0:
                        print("Epoch: " + str(epoch) + "; Loss: " + str(loss_val))

                dataset.reset_pointer()
            end = time.time()
            tot_time += (end - start)

    img_per_sec = (dataset.data_size * num_epochs) / tot_time
    print("Throughout: " + str(img_per_sec) + " images / sec")


if __name__ == '__main__':
    main()

