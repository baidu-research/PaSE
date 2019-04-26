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


def GetDim(dim, name):
    if isinstance(dim, int):
        return mtf.Dimension(name, dim)
    return dim


def GetDim(dim):
    random_name = ''.join([random.choice(string.ascii_letters + string.digits)
        for n in xrange(8)])
    return GetDim(dim, random_name)


class ConvDims():
    def __init__(self, b, c, h, w, n, r, s, name):
        Dim = lambda suffix, dim : GetDim(dim, '%s_%s_dim' % (name, suffix))

        self.b = Dim('b', b)
        self.c = Dim('c', c)
        self.h = Dim('h', h)
        self.w = Dim('w', w)
        self.n = Dim('n', n)
        self.r = Dim('r', r)
        self.s = Dim('s', s)

        self.TensorShape = mtf.Shape([self.b, self.h, self.w. self.c])
        self.KernelShape = mtf.Shape([self.r, self.s, self.c, self.n])


class FCDims():
    def __init__(self, m, n, k, name):
        Dim = lambda suffix, dim : GetDim(dim, '%s_%s_dim' % (name, suffix))

        self.m = Dim('m', m)
        self.n = Dim('n', n)
        self.k = Dim('k', k)

        self.TensorShape = mtf.Shape([self.m, self.k])
        self.KernelShape = mtf.Shape([self.k, self.n])


def Conv2d(tsr, filter_h, filter_w, num_filters, stride_h, stride_w, padding,
        use_bias=True, activation=None, name=None):
    with tf.variable_scope(name, default_name='conv2d'):
        r_dim = mtf.Dimension('%s_r_dim' % name, filter_h)
        s_dim = mtf.Dimension('%s_s_dim' % name, filter_w)
        c_dim = tsr.shape.dims[-1]
        n_dim = mtf.Dimension('%s_n_dim' % name, num_filters)

        w_shape = mtf.Shape([r_dim, s_dim, c_dim, n_dim])
        w = mtf.get_variable(tsr.mesh, 'weight', w_shape, dtype=tsr.dtype)
        out = mtf.conv2d(tsr, w, (1, stride_h, stride_w, 1), padding)

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
        out = mtf.slicewise(max_pool, tsr, output_shape, tsr.dtype, splittable_dims)
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
    print(tf_x.shape)
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
    conv2_dims = ConvDims(x_batch_dim, conv1_dims.n, 27, 27, 256, 5, 5, 'conv2')
    conv3_dims = ConvDims(x_batch_dim, conv2_dims.n, 13, 13, 384, 3, 3, 'conv3')
    conv4_dims = ConvDims(x_batch_dim, conv3_dims.n, conv3_dims.h, conv3_dims.w,
            384, 3, 3, 'conv4')
    conv5_dims = ConvDims(x_batch_dim, conv4_dims.n, conv4_dims.h, conv4_dims.w,
            256, 3, 3, 'conv5')

    # FC layer dimensions
    fc6_dims = FCDims(y_batch_dim, 4096, 9216, name='fc6')
    fc7_dims = FCDims(fc6_dims.m, 4096, 4096, name='fc7')
    fc8_dims = FCDims(fc7_dims.m, y_class_dim, 4096, name='fc8')
    softmax_dim = fc8_dims.n

    def AssignLayout(ta_axes, mesh_axis):
        layout = []
        for a in ta_axes:
            layout.append((a, mesh_axis))
        return layout

    # mtf 1D mesh
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_shape_1d = [('p1', 4),]
    devices = ['gpu:%d' % i for i in range(4)]
    layout = [('conv_batch_dim', 'p1')]
    mesh1_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape_1d,
            layout, devices)

    # mtf 2D mesh
    mesh2 = mtf.Mesh(graph, 'mesh2')
    mesh_shape_2d = [('p1', 4), ('p2', 2)]
    devices = ['gpu:%d' % i for i in range(8)]
    p1_layout = AssignLayout([conv1_dims.b.name, fc6_dims.n.name,
        fc7_dims.n.name, fc8_dims.n.name], 'p1')
    p2_layout = AssignLayout([conv2_dims.n.name, conv3_dims.n.name,
        conv4_dims.n.name, conv5_dims.n.name, fc6_dims.k, fc7_dims.k,
        fc8_dims.k], 'p2')
    layout = p1_layout + p2_layout
    mesh2_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape_2d, layout,
            devices)

    meshes = {mesh1:mesh1_impl, mesh2:mesh2_impl}

    # mtf input / output variables
    mtf_x = mtf.import_tf_tensor(mesh1_impl, mtf.Shape(conv1_dims.TensorShape))
    mtf_y = mtf.import_tf_tensor(mesh2_impl, mtf.Shape([y_batch_dim, y_class_dim]))

    # Model
    with tf.variable_scope('alexnet'):
        conv1 = Conv2d(mtf_x, 11, 11, 96, 4, 4, 'VALID', activation=mtf.relu,
                name='conv1')
        pool1 = MaxPool(conv1, 3, 3, 2, 2, 'VALID', conv2_dims.TensorShape,
                'pool1')

        conv2 = Conv2d(pool1, 5, 5, 256, 1, 1, 'SAME', activation=mtf.relu,
                name='conv2')
        pool2 = MaxPool(conv2, 3, 3, 2, 2, 'VALID', conv3_dims.TensorShape,
                'pool2')

        conv3 = Conv2d(pool2, 3, 3, 384, 1, 1, 'SAME', activation=mtf.relu,
                name='conv3')

        conv4 = Conv2d(conv3, 3, 3, 384, 1, 1, 'SAME', activation=mtf.relu,
                name='conv4')

        conv5 = Conv2d(conv4, 3, 3, 256, 1, 1, 'SAME', activation=mtf.relu,
                name='conv5')
        conv5_out_shape = mtf.Shape([conv5_dims.b, GetDim(6), GetDim(6),
            conv5_dims.n])
        pool5 = MaxPool(conv5, 3, 3, 2, 2, 'VALID', conv5_out_shape, 'pool5')

        flattened = mtf.reshape(pool5, mtf.Shape([fc6_dims.m, fc6_dims.k]))
        fc6 = mtf.dropout(mtf.dense(flattened, fc6_dims.n, activation=mtf.relu,
            name='fc6'), keep_prob)
        fc7 = mtf.dropout(mtf.dense(fc6, fc7_dims.n, activation=mtf.relu,
            name='fc7'), keep_prob)
        fc8 = mtf.dropout(mtf.dense(fc7, fc8_dims.n, name='fc8'), keep_prob)

    # Softmax cross-entropy loss
    with tf.variable_scope('loss'):
        one_hot_labels = mtf.one_hot(labels, y_class_dim)
        mtf_cross_ent = mtf.softmax_cross_entropy_with_logits(fc8, one_hot_labels,
                y_class_dim)
        mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    # Optimizer
    with tf.variable_scope('optimize'):
        grads = mtf.gradients([mtf_loss], [v.outputs[0] for v in
            graph.trainable_variables])
        opt = mtf.optimize.SgdOptimizer(lr=lr)
        grad_updates = opt.apply_grads(grads, graph.trainable_variables)

    # Lowering
    lowering = mtf.Lowering(graph, meshes)
    tf_loss = lowering.export_to_tf_tensor(mtf_loss)
    tf_grad_updates = [lowering.lowered_operation(op) for op in grad_updates]

    # Training
    with tf.name_scope('train'):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(num_epochs):
                dataset.reset_pointer()
                step = 0

                while step < train_batches_per_epoch:
                    loss_val, *rem = sess.run([tf_loss] + tf_grad_updates)

                    if step % display_step == 0 and step > 0:
                        print("Epoch: " + str(epoch) + "; Loss: " + str(loss_val))


if __name__ == '__main__':
    main()

