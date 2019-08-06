import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import string, random

import utils
from utils import GetMeshImpl
import mesh_transformations as mt


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


def Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr):
    lowering = mtf.Lowering(graph, mesh_to_impl)
    out_tsr = lowering.export_to_tf_tensor(mtf_out_tsr)
    assert_op = tf.assert_equal(in_tsr, out_tsr)

    with tf.Session() as sess:
        sess.run(assert_op)


def Concat1(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh_to_impl = {mesh0:GetMeshImpl([4])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([('axis0', shape[0]), *shape[1:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    new_shape = mtf_shape.rename_dimension('axis0', RandName())
    mtf_out_tsr = mtf.reshape(mtf_in_tsr, new_shape)
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def Concat2(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh_to_impl = {mesh0:GetMeshImpl([4], [0, 1, 4, 5])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([shape[0], ('axis0', shape[1]), *shape[2:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    new_shape = mtf_shape.rename_dimension('axis0', RandName())
    mtf_out_tsr = mtf.reshape(mtf_in_tsr, new_shape)
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def Concat3(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh_to_impl = {mesh0:GetMeshImpl([1], [0])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([shape[0], ('axis0', shape[1]), *shape[2:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    new_shape = mtf_shape.rename_dimension('axis0', RandName())
    mtf_out_tsr = mtf.reshape(mtf_in_tsr, new_shape)
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def Concat4(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh_to_impl = {mesh0:GetMeshImpl([2], [0, 4])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([shape[0], ('axis0', shape[1]), *shape[2:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    new_shape = mtf_shape.rename_dimension('axis0', RandName())
    mtf_out_tsr = mtf.reshape(mtf_in_tsr, new_shape)
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def Reduce1(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh_to_impl = {mesh0:GetMeshImpl([4])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([('axis0', shape[0]), *shape[1:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mtf.reduce_sum(mtf_in_tsr, reduced_dim=mtf_shape[0])
    Run(graph, mesh_to_impl, tf.reduce_sum(in_tsr, axis=0), mtf_out_tsr)


def Reduce2(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh_to_impl = {mesh0:GetMeshImpl([4], [0, 1, 4, 5])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([shape[0], ('axis0', shape[1]), *shape[2:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mtf.reduce_sum(mtf_in_tsr, reduced_dim=mtf_shape[1])
    Run(graph, mesh_to_impl, tf.reduce_sum(in_tsr, axis=1), mtf_out_tsr)


def Reduce3(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh_to_impl = {mesh0:GetMeshImpl([4], [0, 1, 4, 5])}

    in_tsr = in_tsr[:,:,0,0]
    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([shape[0], ('axis0', shape[1]), *shape[2:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mtf.reduce_sum(mtf_in_tsr, reduced_dim=mtf_shape[1])
    Run(graph, mesh_to_impl, tf.reduce_sum(in_tsr, axis=1), mtf_out_tsr)


def Reduce4(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh_to_impl = {mesh0:GetMeshImpl([1], [0])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([shape[0], ('axis0', shape[1]), *shape[2:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mtf.reduce_sum(mtf_in_tsr, reduced_dim=mtf_shape[1])
    Run(graph, mesh_to_impl, tf.reduce_sum(in_tsr, axis=1), mtf_out_tsr)


def Reduce5(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh_to_impl = {mesh0:GetMeshImpl([2], [0, 4])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([shape[0], ('axis0', shape[1]), *shape[2:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mtf.reduce_sum(mtf_in_tsr, reduced_dim=mtf_shape[1])
    Run(graph, mesh_to_impl, tf.reduce_sum(in_tsr, axis=1), mtf_out_tsr)


def main():
    ndims = 4
    shape = [16, 32, 64, 128]
    in_tsr = tf.constant(np.random.randint(128, size=shape), shape=shape,
            verify_shape=True, dtype=tf.float32)

    Concat1(in_tsr)
    Concat2(in_tsr)
    Concat3(in_tsr)
    Concat4(in_tsr)
    Reduce1(in_tsr)
    Reduce2(in_tsr)
    Reduce3(in_tsr)
    Reduce4(in_tsr)
    Reduce5(in_tsr)

    print('Tests passed.')


if __name__ == '__main__':
    main()
