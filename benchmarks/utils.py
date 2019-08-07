from operator import mul
from functools import reduce

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf
#import dgx_mesh_impl


def Prod(lst):
    return reduce(mul, lst, 1)


def GetDeviceList(gpus):
    if isinstance(gpus, list) and all(
            isinstance(gpu, tf.DeviceSpec) for gpu in gpus):
        return gpus

    gpus = range(gpus) if isinstance(gpus, int) else gpus
    return [tf.DeviceSpec(device_type='GPU', device_index=i) for i in gpus]


def AssignLayout(ta_axes, mesh_axis):
    layout = []
    for a in ta_axes:
        a = a.name if isinstance(a, mtf.Dimension) else a
        layout.append((a, mesh_axis))
    return layout


def RenameDim(shape, axis, name):
    assert isinstance(shape, mtf.Shape)
    return shape.rename_dimension(shape[axis].name, name)


def RenameDims(shape, axes, names):
    assert len(axes) == len(names)
    for axis, name in zip(axes, names):
        shape = RenameDim(shape, axis, name)
    return shape


def GetMeshImpl(dev_cnts, devices=None, axes=None):
    axes = axes or ['axis%d' % i for i in range(len(dev_cnts))]
    assert len(dev_cnts) == len(axes)

    mesh_shape = []
    layout_rules = []
    for d, axis in zip(dev_cnts, axes):
        mesh_shape.append((axis, d))
        layout_rules.append((axis, axis))

    devices = GetDeviceList(devices or Prod(dev_cnts))
    #return dgx_mesh_impl.DGXMeshImpl(mesh_shape, layout_rules, devices)
    return mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules,
            devices)


'''
def GetMeshImpls(graph, devices_list):
    meshes = []
    mesh_to_impl = {}

    for i, devices in enumerate(devices_list):
        mesh = mtf.Mesh(graph, 'mesh_%d' % i)
        meshes.append(mesh)
        mesh_to_impl[mesh] = GetMeshImpl(devices)

    return meshes, mesh_to_impl
'''


# Converts 'v' into a tuple (v, v) if 'v' is a scalar
def MakePair(v):
    if hasattr(v, "__len__"):
        assert len(v) == 2
        return v
    else:
        return (v, v)


def FlattenList(l):
   return [item for sublist in l for item in sublist]


def TransposeLists(l):
    return [list(x) for x in zip(*l)]


'''
class ReplaceMeshOperation(mtf.Operation):
    def __init__(self, new_mesh, input, axis, lowering_fn=None, name=None):
        assert isinstance(axis, mtf.Dimension)
        super().__init__([input], mesh=new_mesh, name=name or 'replace_mesh')
        self.old_mesh = input.mesh
        self.axis = axis
        self._outputs = [mtf.Tensor(self, input.shape, input.dtype)]

        self.lowering_fn = lowering_fn

    def lower(self, lowering):
        if self.lowering_fn is not None:
            input_slices = \
                    lowering.tensors[self.inputs[0]].to_laid_out_tensor().tensor_list
            output_slices = self.lowering_fn(input_slices, self.old_mesh,
                    self.mesh, self.axis)

            laid_out_tensor = \
                    lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(output_slices)
            lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)

        else:
            raise NotImplementedError('Lowering not implemented.')
'''


class ReplaceMeshWithRemovalOperation(mtf.Operation):
    def __init__(self, new_mesh, input, axis, name=None):
        assert isinstance(axis, mtf.Dimension)
        super().__init__([input], mesh=new_mesh, name=name or
                'replace_mesh_with_removal')
        self.old_mesh = input.mesh
        self.axis = axis
        self._outputs = [mtf.Tensor(self, input.shape, input.dtype)]

    def gradient(self, grad_ys):
        return ReplaceMeshWithReplicationOperation(self.old_mesh, grad_ys[0],
                self.axis).outputs

    def lower(self, lowering):
        input_slices = \
                lowering.tensors[self.inputs[0]].to_laid_out_tensor().tensor_list

        mesh_impl = lowering.mesh_impl(self.old_mesh)
        axis_num = mesh_impl.shape.dims.index(self.axis)
        cumprod = mesh_impl.shape.cumprod[axis_num]

        # Make sure the mesh axes are compatible
        old_dims = mesh_impl.shape.to_integer_list
        new_dims = lowering.mesh_impl(self.mesh).shape.to_integer_list
        assert len(old_dims) == len(new_dims) + 1
        assert old_dims[:axis_num] == new_dims[:axis_num]
        assert old_dims[axis_num+1:] == new_dims[axis_num:]

        # Make sure the tensor is replicated along 'axis_num' mesh dimension
        tsr_layout = mesh_impl.tensor_layout(self.inputs[0])
        assert tsr_layout.mesh_axis_to_tensor_axis(axis_num+1)[-1] is None

        output_slices = [input_slices[i:i+cumprod] for i in range(0,
            len(input_slices), cumprod * self.axis.size)]
        output_slices = FlattenList(output_slices)
        #output_slices = [tf.identity(s, name='replicated_%d' % i) for i, s in
        #        enumerate(output_slices)]

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


class ReplaceMeshWithReplicationOperation(mtf.Operation):
    def __init__(self, new_mesh, input, axis, name=None):
        assert isinstance(axis, mtf.Dimension)
        assert axis.name not in input.shape.dimension_names

        super().__init__([input], mesh=new_mesh, name=name or
                'replace_mesh_with_replication')
        self.old_mesh = input.mesh
        self.axis = axis
        self._outputs = [mtf.Tensor(self, input.shape, input.dtype)]

    def gradient(self, grad_ys):
        return ReplaceMeshWithRemovalOperation(self.old_mesh, grad_ys[0],
                self.axis).outputs

    def lower(self, lowering):
        input_slices = \
                lowering.tensors[self.inputs[0]].to_laid_out_tensor().tensor_list

        mesh_impl = lowering.mesh_impl(self.mesh)
        axis_num = mesh_impl.shape.dims.index(self.axis)
        cumprod = mesh_impl.shape.cumprod[axis_num]

        # Make sure the mesh axes are compatible
        old_dims = lowering.mesh_impl(self.old_mesh).shape.to_integer_list
        new_dims = mesh_impl.shape.to_integer_list
        assert len(old_dims) == len(new_dims) - 1
        assert old_dims[:axis_num] == new_dims[:axis_num]
        assert old_dims[axis_num:] == new_dims[axis_num+1:]

        output_slices = [input_slices[i:i+cumprod] * self.axis.size for i in
                range(0, len(input_slices), cumprod)]
        output_slices = FlattenList(output_slices)
        #output_slices = [tf.identity(s, name='replicated_%d' % i) for i, s in
        #        enumerate(output_slices)]

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


class ReplaceMeshWithConcatOperation(mtf.Operation):
    def __init__(self, new_mesh, input, axis, name=None):
        assert isinstance(axis, mtf.Dimension)
        assert input.mesh.shape.dimension_names == new_mesh.shape.dimension_names

        super().__init__([input], mesh=new_mesh, name=name or
                'replace_mesh_with_concat')
        self.old_mesh = input.mesh
        self.axis = axis
        self._outputs = [mtf.Tensor(self, input.shape, input.dtype)]

    def gradient(self, grad_ys):
        return ReplaceMeshWithSplitOperation(self.old_mesh, grad_ys[0],
                self.axis).outputs

    def lower(self, lowering):
        input_slices = \
                lowering.tensors[self.inputs[0]].to_laid_out_tensor().tensor_list

        mesh_impl = lowering.mesh_impl(self.old_mesh)
        axis_num = mesh_impl.shape.dims.index(self.axis)
        cumprod = mesh_impl.shape.cumprod[axis_num]

        concat_slices = []
        for i in range(0, len(input_slices), cumprod * self.axis.size):
            slices = []
            for j in range(i, i + cumprod * self.axis.size, cumprod):
                slices.append(input_slices[j:j+cumprod])
            concat_slices.append(TransposeLists(slices))

        output_slices = []
        for i, s in enumerate(concat_slices):
            with tf.device(s[0]):
                output_slices.append(tf.concat(s, axis=axis_num,
                    name='concat_%d' % i))

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)



class ReplaceMeshWithSplitOperation(mtf.Operation):
    def __init__(self, new_mesh, input, axis, name=None):
        assert isinstance(axis, mtf.Dimension)
        assert input.mesh.shape.dimension_names == new_mesh.shape.dimension_names

        super().__init__([input], mesh=new_mesh, name=name or
                'replace_mesh_with_split')
        self.old_mesh = input.mesh
        self.axis = axis
        self._outputs = [mtf.Tensor(self, input.shape, input.dtype)]

    def gradient(self, grad_ys):
        return ReplaceMeshWithConcatOperation(self.old_mesh, grad_ys[0],
                self.axis).outputs

    def lower(self, lowering):
        input_slices = \
                lowering.tensors[self.inputs[0]].to_laid_out_tensor().tensor_list

        mesh_impl = lowering.mesh_impl(self.mesh)
        axis_num = mesh_impl.shape.dims.index(self.axis)
        cumprod = mesh_impl.shape.cumprod[axis_num]

        split_slices = []
        for i, s in enumerate(input_slices):
            with tf.device(s.device):
                split_slices.append(tf.split(s, self.axis.size,
                    axis=axis_num, name='split_%d' % i))
        split_slices = TransposeLists(split_slices)

        output_slices = []
        for i in range(0, len(input_slices), cumprod):
            for s in split_slices:
                output_slices.append(s[i:i+cumprod])

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


'''
def ReplaceMesh(new_mesh, tsr, axis, lowering_fn, name=None):
    return ReplaceMeshOperation(new_mesh, tsr, axis, lowering_fn,
            name=name).outputs[0]
'''


def ReplaceMeshWithReplication(new_mesh, tsr, axis, name=None):
    return ReplaceMeshWithReplicationOperation(new_mesh, tsr, axis,
            name).outputs[0]


def ReplaceMeshWithRemoval(new_mesh, tsr, axis, name=None):
    return ReplaceMeshWithRemovalOperation(new_mesh, tsr, axis, name).outputs[0]

