import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import string, random

import utils
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


def Transpose1(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([4, 2]), mesh1:utils.GetMeshImpl([4,2])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([('axis0', shape[0]), ('axis1', shape[1]), *shape[2:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), RandName(), 'axis0', 'axis1'])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def Transpose2(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([4, 2]), \
            mesh1:utils.GetMeshImpl([2, 4])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([('axis0', shape[0]), ('axis1', shape[1]), *shape[2:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), RandName(), 'axis0', 'axis1'])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def Transpose3(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([4, 2]), \
            mesh1:utils.GetMeshImpl([2, 4])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([('axis0', shape[0]), shape[1], ('axis1', shape[2]),
        shape[3]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), 'axis0', RandName(), 'axis1'])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def Transpose4(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([4, 2]), \
            mesh1:utils.GetMeshImpl([2, 4])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([('axis0', shape[0]), shape[1], ('axis1', shape[2]),
        shape[3]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), 'axis1', RandName(), 'axis0'])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def DependentAxes(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([4, 2]), mesh1:utils.GetMeshImpl([4,2])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([('axis0', shape[0]), ('axis1', shape[1]), *shape[2:]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), 'axis0', 'axis1', RandName()])

    try:
        Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)
        assert False # This run should fail and throw ValueError
    except ValueError:
        return


def Broadcast1(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([4, 2]), mesh1:utils.GetMeshImpl([8])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([('axis0', shape[0]), shape[1], ('axis1', shape[2]),
        shape[3]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), 'axis0', RandName(), RandName()])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def Broadcast2(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([4, 2]), mesh1:utils.GetMeshImpl([8])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([('axis0', shape[0]), shape[1], ('axis1', shape[2]),
        shape[3]])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), RandName(), RandName(), RandName()])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def Contract1(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([8]), \
            mesh1:utils.GetMeshImpl([4, 2])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape([*shape[:3], ('axis0', shape[3])])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), 'axis0', 'axis1', RandName()])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def Contract2(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([2, 4]), \
            mesh1:utils.GetMeshImpl([4, 2])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape(shape)
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), 'axis0', 'axis1', RandName()])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def LessDevices1(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([2, 4]), \
            mesh1:utils.GetMeshImpl([4])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape(shape[:-2] + [('axis1', shape[2]), ('axis0', shape[3])])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), 'axis0', RandName(), RandName()])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def LessDevices2(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([2]), \
            mesh1:utils.GetMeshImpl([4])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape(shape[:-1] + [('axis0', shape[3])])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), 'axis0', RandName(), RandName()])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def MoreDevices(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([2]), \
            mesh1:utils.GetMeshImpl([8])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape(shape[:-1] + [('axis0', shape[-1])])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), 'axis0', RandName(), RandName()])
    Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)


def WrongShape(in_tsr):
    graph = mtf.Graph()
    mesh0 = mtf.Mesh(graph, 'mesh0')
    mesh1 = mtf.Mesh(graph, 'mesh1')
    mesh_to_impl = {mesh0:utils.GetMeshImpl([4, 2]), \
            mesh1:utils.GetMeshImpl([8])}

    shape = in_tsr.get_shape().as_list()
    mtf_shape = GetShape(shape[:-2] + [('axis0', shape[2]), ('axis0', shape[3])])
    mtf_in_tsr = mtf.import_tf_tensor(mesh0, in_tsr, mtf_shape)
    mtf_out_tsr = mt.ReplaceMeshWithIndependentAxes(mtf_in_tsr, mesh1,
            [RandName(), 'axis0', RandName(), RandName()])

    try:
        Run(graph, mesh_to_impl, in_tsr, mtf_out_tsr)
        assert False # This test should fail with ValueError
    except ValueError:
        return


def main():
    ndims = 4
    shape = [16, 32, 64, 128]
    in_tsr = tf.constant(np.random.randint(utils.Prod(shape) * 8, size=shape),
            shape=shape, verify_shape=True)

    Transpose1(in_tsr)
    Transpose2(in_tsr)
    Transpose3(in_tsr)
    Transpose4(in_tsr)
    DependentAxes(in_tsr)
    Broadcast1(in_tsr)
    Broadcast2(in_tsr)
    Contract1(in_tsr)
    Contract2(in_tsr)
    LessDevices1(in_tsr)
    LessDevices2(in_tsr)
    MoreDevices(in_tsr)
    WrongShape(in_tsr)
    print('Tests passed.')


if __name__ == '__main__':
    main()

