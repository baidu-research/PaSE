import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import sys
import time
import os
from datetime import datetime
from argparse import ArgumentParser
import functools

from dataloader import ImageDataLoader


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
    dropout_rate = args['dropout']
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
    
    # Input tensors
    tf_x, tf_y = dataset.next_batch()
    print(tf_x.shape)
    tf_x.set_shape([batch_size, 227, 227, 3])
    tf_y.set_shape([batch_size])

    # mtf graph
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, 'mesh')

    def GetDim(dim, name):
        if isinstance(dim, int):
            return mtf.Dimension('%s_%s_dim' % name, dim)
        return dim

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
    conv1_dims = BuildConvDims(x_batch_dim, x_c_dim, x_h_dim, x_w_dim, 96, 4, 4,
            'conv1')
    conv2_dims = BuildConvDims(x_batch_dim, conv1_dims['n'], 27, 27, 256, 5, 5,
            'conv2')
    conv3_dims = BuildConvDims(x_batch_dim, conv2_dims['n'], 13, 13, 384, 3, 3,
            'conv3')
    conv4_dims = BuildConvDims(x_batch_dim, conv3_dims['n'], conv3_dims['h'],
            conv3_dims['w'], 384, 3, 3, 'conv4')
    conv5_dims = BuildConvDims(x_batch_dim, conv4_dims['n'], conv4_dims['h'],
            conv4_dims['w'], 256, 3, 3, 'conv5')

    # FC layer dimensions
    fc6_dims = BuildFCDims(y_batch_dim, 4096, 9216, name='fc6')
    fc7_dims = BuildFCDims(fc6_dims['m'], 4096, 4096, name='fc7')
    fc8_dims = BuildFCDims(fc7_dims['m'], num_classes, 4096, name='fc8')
    softmax_dim = fc8_dims['n']

    def AssignLayout(ta_axes, mesh_axis):
        layout = []
        for a in ta_axes:
            layout.append((a, mesh_axis))
        return layout

    # mtf 1D mesh
    mesh_shape_1d = [('p1', 4),]
    devices = ['gpu:%d' % i for i in range(4)]
    layout = [('conv_batch_dim', 'p1')]
    mesh1 = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape_1d, layout,
            devices)

    # mtf 2D mesh
    mesh_shape_2d = [('p1', 4), ('p2', 2)]
    devices = ['gpu:%d' % i for i in range(8)]
    p1_layout = AssignLayout([conv1_dims['b'].name, fc6_dims['n'].name,
        fc7_dims['n'].name, fc8_dims['n'].name], 'p1')
    p2_layout = AssignLayout([conv2_dims['n'].name, conv3_dims['n'].name,
        conv4_dims['n'].name, conv5_dims['n'].name, fc6_dims['k'],
        fc7_dims['k'], fc8_dims['k']], 'p2')
    layout = p1_layout + p2_layout
    mesh2 = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape_2d, layout,
            devices)

    # mtf input / output variables
    mtf_x = mtf.import_tf_tensor(mesh1, mtf.Shape([x_batch_dim, x_h_dim,
        x_w_dim, x_c_dim]))
    mtf_y = mtf.import_tf_tensor(mesh2, mtf.Shape([y_batch_dim, y_class_dim]))


if __name__ == '__main__':
    main()
