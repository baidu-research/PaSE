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


def GetDim(dim, name):
    if isinstance(dim, int):
        return mtf.Dimension(name, dim)
    return dim


def GetTempDim(dim):
    random_name = ''.join([random.choice(string.ascii_letters + string.digits)
        for n in range(8)])
    return GetDim(dim, random_name)


class ConvDims():
    def __init__(self, b_, c_, h_, w_, n_, r_, s_, name):
        Dim = lambda suffix, dim : GetDim(dim, '%s_%s_dim' % (name, suffix))

        self.b = Dim('b', b_)
        self.c = Dim('c', c_)
        self.h = Dim('h', h_)
        self.w = Dim('w', w_)
        self.n = Dim('n', n_)
        self.r = Dim('r', r_)
        self.s = Dim('s', s_)

        self.tensor_shape = mtf.Shape([self.b, self.h, self.w, self.c])
        self.kernel_shape = mtf.Shape([self.r, self.s, self.c, self.n])


class FCDims():
    def __init__(self, m, n, k, name):
        Dim = lambda suffix, dim : GetDim(dim, '%s_%s_dim' % (name, suffix))

        self.m = Dim('m', m)
        self.n = Dim('n', n)
        self.k = Dim('k', k)

        self.tensor_shape = mtf.Shape([self.m, self.k])
        self.kernel_shape = mtf.Shape([self.k, self.n])


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


def Conv2d(tsr, filter_h, filter_w, num_filters, stride_h, stride_w, padding,
        use_bias=True, activation=None, name=None):
    with tf.variable_scope(name, default_name='conv2d'):
        r_dim = mtf.Dimension('%s_r_dim' % name, filter_h)
        s_dim = mtf.Dimension('%s_s_dim' % name, filter_w)
        c_dim = tsr.shape.dims[-1]
        n_dim = mtf.Dimension('%s_n_dim' % name, num_filters)

        w_shape = mtf.Shape([r_dim, s_dim, c_dim, n_dim])
        w = mtf.get_variable(tsr.mesh, 'weight', w_shape, dtype=tsr.dtype)
        out = Conv2dOperation(tsr, w, (1, stride_h, stride_w, 1),
                padding).outputs[0]

        if use_bias:
            b = mtf.get_variable(tsr.mesh, 'bias', mtf.Shape([n_dim]),
                    initializer=tf.zeros_initializer(), dtype=tsr.dtype)
            out += b

        if activation:
            out = activation(out)

        return out


def MaxPool(tsr, filter_h, filter_w, stride_h, stride_w, padding, output_shape, name):
    with tf.variable_scope(name):
        def max_pool(x):
            return tf.nn.max_pool(x, [1, filter_h, filter_w, 1], [1, stride_h,
                stride_w, 1], padding)

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

    # mtf graph
    graph = mtf.Graph()

    def BuildConvDims(b, c, h, w, n, r, s, name):
        Dim = lambda suffix, dim : GetDim(dim, '%s_%s_dim' % (name, suffix))
        return {'b':Dim('b', b), 'c':Dim('c', c), 'h':Dim('h', h), 'w':Dim('w',
            w), 'n':Dim('n', n), 'r':Dim('r', r), 's':Dim('s', s)}

    def BuildFCDims(m, n, k, name):
        Dim = lambda suffix, dim : GetDim(dim, '%s_%s_dim' % (name, suffix))
        return {'m':Dim('m', m), 'n':Dim('n', n), 'k':Dim('k', k)}

    # Input dimensions
    x_batch_dim = mtf.Dimension('x_batch_dim', batch_size)
    x_c_dim = mtf.Dimension('x_c_dim', 3)
    x_h_dim = mtf.Dimension('x_h_dim', 227)
    x_w_dim = mtf.Dimension('x_w_dim', 227)
    y_batch_dim = mtf.Dimension('y_batch_dim', batch_size)
    y_class_dim = mtf.Dimension('y_class_dim', num_classes)

    # Conv layer dimensions
    conv1_dims = ConvDims(x_batch_dim, x_c_dim, x_h_dim, x_w_dim, 96, 4, 4,
            'conv1')
    conv2_dims = ConvDims(x_batch_dim, conv1_dims.n.size, GetDim(27,
        x_h_dim.name), GetDim(27, x_w_dim.name), 256, 5, 5, 'conv2')
    conv3_dims = ConvDims(x_batch_dim, conv2_dims.n.size, GetDim(13,
        x_h_dim.name), GetDim(13, x_w_dim.name), 384, 3, 3, 'conv3')
    conv4_dims = ConvDims(x_batch_dim, conv3_dims.n.size, GetDim(13,
        x_h_dim.name), GetDim(13, x_w_dim.name), 384, 3, 3, 'conv4')
    conv5_dims = ConvDims(x_batch_dim, conv4_dims.n.size, GetDim(13,
        x_h_dim.name), GetDim(13, x_w_dim.name), 256, 3, 3, 'conv5')

    # Flattened dimensions
    before_flattened_h = GetTempDim(6)
    before_flattened_w = GetTempDim(6)
    after_flattened_h = GetTempDim(1)
    after_flattened_w = GetTempDim(1)

    # FC layer dimensions
    fc6_dims = FCDims(y_batch_dim, 4096, 9216, name='fc6')
    fc7_dims = FCDims(y_batch_dim, 4096, 4096, name='fc7')
    fc8_dims = FCDims(y_batch_dim, y_class_dim, 4096, name='fc8')
    softmax_dim = fc8_dims.n

    def AssignLayout(ta_axes, mesh_axis):
        layout = []
        for a in ta_axes:
            layout.append((a, mesh_axis))
        return layout

    # mtf 1D mesh
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_shape_1d = [('m1_p1', 4),]
    devices = ['gpu:%d' % i for i in range(4)]
    layout = [('x_batch_dim', 'm1_p1')]
    mesh1_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape_1d,
            layout, devices)

    # mtf 2D mesh
    mesh2 = mtf.Mesh(graph, 'mesh2')
    mesh_shape_2d = [('m2_p1', 4), ('m2_p2', 2), ('m2_h', 1), ('m2_w', 1)]
    devices = ['gpu:%d' % i for i in range(8)]
    p1_layout = AssignLayout([x_batch_dim.name, fc6_dims.n.name,
        fc7_dims.n.name, fc8_dims.n.name], 'm2_p1')
    p2_layout = AssignLayout([conv2_dims.n.name, conv3_dims.n.name,
        conv4_dims.n.name, conv5_dims.n.name, fc6_dims.k.name, fc7_dims.k.name,
        fc8_dims.k.name], 'm2_p2')
    h_layout = AssignLayout([before_flattened_h.name, after_flattened_h.name], 'm2_h')
    w_layout = AssignLayout([before_flattened_w.name, after_flattened_w.name], 'm2_w')
    layout = p1_layout + p2_layout + h_layout + w_layout
    mesh2_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape_2d, layout,
            devices)

    meshes = {mesh1:mesh1_impl, mesh2:mesh2_impl}

    # mtf input / output variables
    mtf_x = mtf.import_tf_tensor(mesh1, tf_x, conv1_dims.tensor_shape)
    mtf_y = mtf.import_tf_tensor(mesh2, tf_y, mtf.Shape([y_batch_dim]))

    # Model
    with tf.variable_scope('alexnet'):
        conv1 = Conv2d(mtf_x, 11, 11, 96, 4, 4, 'VALID', activation=mtf.relu,
                name='conv1')
        pool1_shape = mtf.Shape([conv1_dims.b, conv2_dims.h, conv2_dims.w,
            conv1_dims.n])
        pool1 = MaxPool(conv1, 3, 3, 2, 2, 'VALID', pool1_shape, 'pool1')

        #slice1 = mtf.slice(pool1, 0, 48, conv1_dims.n.name, name='slice1')
        #slice2 = mtf.slice(pool1, 48, 48, conv1_dims.n.name, name='slice2')

        lowering = mtf.Lowering(graph, meshes)
        #slice1_all_slices = lowering.tensors[slice1].to_laid_out_tensor().all_slices
        #slice2_all_slices = lowering.tensors[slice2].to_laid_out_tensor().all_slices
        #laid_out_pool1 = FlattenList(TransposeLists([slice1_all_slices,
        #    slice2_all_slices]))
        #laid_out_pool1 = mesh2_impl.LaidOutTensor(laid_out_pool1)
        pool1_slices = lowering.tensors[pool1].to_laid_out_tensor().all_slices
        laid_out_pool1 = \
                mesh2_impl.LaidOutTensor(FlattenList(TransposeLists([pool1_slices,
                    pool1_slices])))

        pool1_mesh2 = mtf.import_laid_out_tensor(mesh2, laid_out_pool1,
                conv2_dims.tensor_shape, name='pool1_mesh2')

        conv2 = Conv2d(pool1_mesh2, 5, 5, 256, 1, 1, 'SAME',
                activation=mtf.relu, name='conv2')
        pool2_shape = mtf.Shape([conv2_dims.b, conv3_dims.h, conv3_dims.w,
            conv2_dims.n])
        pool2 = MaxPool(conv2, 3, 3, 2, 2, 'VALID', pool2_shape, 'pool2')
        pool2 = mtf.rename_dimension(pool2, pool2.shape[-1].name,
                conv3_dims.c.name)

        conv3 = Conv2d(pool2, 3, 3, 384, 1, 1, 'SAME', activation=mtf.relu,
                name='conv3')
        conv3 = mtf.rename_dimension(conv3, conv3.shape[-1].name,
                conv4_dims.c.name)

        conv4 = Conv2d(conv3, 3, 3, 384, 1, 1, 'SAME', activation=mtf.relu,
                name='conv4')
        conv4 = mtf.rename_dimension(conv4, conv4.shape[-1].name,
                conv5_dims.c.name)

        conv5 = Conv2d(conv4, 3, 3, 256, 1, 1, 'SAME', activation=mtf.relu,
                name='conv5')
        pool5_shape = mtf.Shape([conv5_dims.b, GetDim(6, x_h_dim.name),
            GetDim(6, x_w_dim.name), conv5_dims.n])
        pool5 = MaxPool(conv5, 3, 3, 2, 2, 'VALID', pool5_shape, 'pool5')

        flattened = mtf.reshape(pool5, mtf.Shape([conv5_dims.b,
            before_flattened_h, before_flattened_w, conv5_dims.n]),
            name='flattening1')
        flattened = mtf.reshape(flattened, mtf.Shape([fc6_dims.m, fc6_dims.k,
            after_flattened_h, after_flattened_w]), name='flattening2')
        flattened = mtf.reshape(flattened, mtf.Shape([fc6_dims.m, fc6_dims.k]),
                name='flattening3')

        fc6 = mtf.dropout(mtf.layers.dense(flattened, fc6_dims.n,
            activation=mtf.relu, name='fc6'), keep_prob)
        fc6 = mtf.rename_dimension(fc6, fc6.shape[-1].name, fc7_dims.k.name)
        fc7 = mtf.dropout(mtf.layers.dense(fc6, fc7_dims.n, activation=mtf.relu,
            name='fc7'), keep_prob)
        fc7 = mtf.rename_dimension(fc7, fc7.shape[-1].name, fc8_dims.k.name)
        fc8 = mtf.dropout(mtf.layers.dense(fc7, fc8_dims.n, name='fc8'), keep_prob)

    # Softmax cross-entropy loss
    with tf.variable_scope('loss'):
        one_hot_labels = mtf.one_hot(mtf_y, y_class_dim)
        mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(fc8,
                one_hot_labels, y_class_dim)
        mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    # Optimizer
    with tf.variable_scope('optimize'):
        grads = mtf.gradients([mtf_loss], [v.outputs[0] for v in
            graph.trainable_variables])
        opt = mtf.optimize.SgdOptimizer(lr=learning_rate)
        grad_updates = opt.apply_grads(grads, graph.trainable_variables)

    # Lowering
    lowering = mtf.Lowering(graph, meshes)
    tf_loss = lowering.export_to_tf_tensor(mtf_loss)
    tf_grad_updates = [lowering.lowered_operation(op) for op in grad_updates]

    # Training
    with tf.name_scope('train'):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):
                dataset.reset_pointer()
                step = 0

                while step < train_batches_per_epoch:
                    loss_val, *rem = sess.run([tf_loss] + tf_grad_updates)

                    if step % display_step == 0 and step > 0:
                        print("Epoch: " + str(epoch) + "; Loss: " + str(loss_val))


if __name__ == '__main__':
    main()

