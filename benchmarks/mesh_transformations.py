from operator import mul
from functools import reduce

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import utils


class MeshReplacementOperation(mtf.Operation):
    def __init__(self, x, mesh, dim_names=None, name=None):
        assert x.mesh != mesh

        self.old_mesh = x.mesh
        if isinstance(dim_names, mtf.Shape):
            dim_names = dim_names.dimension_names
        self.new_dim_names = dim_names or x.shape.dimension_names

        assert len(self.new_dim_names) == len(x.shape)
        self.new_shape = mtf.Shape([mtf.Dimension(name or dim.name, dim.size)
            for name, dim in zip(self.new_dim_names, x.shape.dims)])

        super().__init__([x], mesh=mesh, name=name or self.__class__.__name__)
        self._outputs = [mtf.Tensor(self, self.new_shape, x.dtype)]

    def gradient(self, grad_ys):
        return self.__class__(grad_ys[0], self.old_mesh,
                self.inputs[0].shape.dimension_names).outputs


# Simple mesh replacement. Each processor is expected to contain same slice in
# both old and new meshes.
class ReplaceMeshOperation(MeshReplacementOperation):
    def lower(self, lowering):
        x = self.inputs[0]
        input_slices = lowering.tensors[x].to_laid_out_tensor().tensor_list
        old_mesh_impl = lowering.mesh_impl(self.old_mesh)
        new_mesh_impl = lowering.mesh_impl(self)

        assert old_mesh_impl.shape.size == new_mesh_impl.shape.size

        # Make sure the devices are in the same order
        assert old_mesh_impl.devices == new_mesh_impl.devices

        # Make sure the slice shape is same in old and new mesh
        assert (old_mesh_impl.slice_shape(x.shape) ==
                new_mesh_impl.slice_shape(self.new_shape))

        # Make sure that each processor in old and new meshes have same slices
        # of the original tensor
        assert all(old_mesh_impl.slice_begin(x.shape, i) ==
                new_mesh_impl.slice_begin(self.new_shape, i) 
                for i in range(old_mesh_impl.shape.size))

        laid_out_tensor = new_mesh_impl.LaidOutTensor(input_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)

def ReplaceMesh(x, mesh, dim_names=None, name=None):
    return ReplaceMeshOperation(x, mesh, dim_names, name).outputs[0]


# Replace mesh by adding/removing duplicate slices
class ReplaceMeshWithDuplicatesOperation(MeshReplacementOperation):
    def lower(self, lowering):
        x = self.inputs[0]
        input_slices = lowering.tensors[x].to_laid_out_tensor().tensor_list

        old_mesh_impl = lowering.mesh_impl(self.old_mesh)
        new_mesh_impl = lowering.mesh_impl(self)
        old_mesh_shape = old_mesh_impl.shape
        new_mesh_shape = new_mesh_impl.shape
        old_devices = old_mesh_impl.devices
        new_devices = new_mesh_impl.devices
        old_num_gpus = old_mesh_impl.shape.size
        new_num_gpus = new_mesh_impl.shape.size
        replicate = (new_num_gpus > old_num_gpus)

        # Get the axes along which slices need to be removed/replicated
        axes = []
        assert old_mesh_impl.ndims == new_mesh_impl.ndims
        old_ma2ta = old_mesh_impl.tensor_layout(x).mesh_axis_to_tensor_axis(
                old_mesh_impl.ndims)
        new_ma2ta = new_mesh_impl.tensor_layout(x).mesh_axis_to_tensor_axis(
                new_mesh_impl.ndims)
        for i, (dim1, dim2) in enumerate(zip(old_mesh_shape, new_mesh_shape)):
            if dim1.size == dim2.size:
                continue

            # Make sure the tensor is not split along the replication axis
            assert (old_ma2ta[i] == new_ma2ta[i] == None)
            if replicate:
                assert dim2.size > dim1.size
            else:
                assert dim1.size > dim2.size

            axes.append(i)

        # Get groups of processors that have same duplicate slices
        old_groups = mtf.processor_groups(old_mesh_shape, axes)
        new_groups = mtf.processor_groups(new_mesh_shape, axes)
        assert len(old_groups) == len(new_groups)

        # Replicate/remove slices within each group
        output_slices = [None] * new_num_gpus
        for old_group, new_group in zip(old_groups, new_groups):
            assert all(old_devices[i] == new_devices[j] for i, j in
                    zip(old_group, new_group))

            if replicate:
                assert len(new_group) % len(old_group) == 0
                ratio = len(new_group) // len(old_group)
                old_group *= ratio

                for old_pnum, new_pnum in zip(old_group, new_group):
                    with tf.device(new_devices[new_pnum]):
                        output_slices[new_pnum] = tf.identity(
                                input_slices[old_pnum])

            else:
                for old_pnum, new_pnum in zip(old_group, new_group):
                    assert (old_devices[old_pnum] == new_devices[new_pnum])
                    assert ((not input_slices[old_pnum].device) or
                            (input_slices[old_pnum].device ==
                                new_devices[new_pnum]))
                    output_slices[new_pnum] = input_slices[old_pnum]

        assert all(s is not None for s in output_slices)
        laid_out_tensor = new_mesh_impl.LaidOutTensor.from_tensor_list(
                output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)

def ReplaceMeshWithDuplicates(x, mesh, dim_names=None, name=None):
    return ReplaceMeshWithDuplicatesOperation(x, mesh, dim_names,
            name).outputs[0]

# Modification of mtf.combine_slices. More communication efficient than
# mtf.combine_slices for our cases. Picks the slices from appropriate group when
# 'tensor_axis' is none.
def combine_slices(slices, tensor_shape, mesh_impl, pnum):
    if tensor_shape.ndims == 0:
        return slices[0]

    def groups_generator():
        mesh_shape = mesh_impl.shape
        ndims = mesh_impl.ndims
        mesh_size = mesh_shape.size
        pnums = list(range(mesh_size))
        for i in range(ndims):
            axes = [j for j in range(ndims) if j != i]
            group_ids = {p:mtf.pnum_to_group(mesh_shape, axes, p)
                    for p in pnums}

            group_id = group_ids[pnum % mesh_size]
            pnums = [p for p,g in group_ids.items() if g == group_id]
            devices = [mesh_impl.devices[p] for p in pnums]

            yield group_id, devices

    ret = slices[:]
    assert len(ret) == mesh_impl.size
    tensor_layout = mesh_impl.tensor_layout(tensor_shape)
    tensor_axes = tensor_layout.mesh_axis_to_tensor_axis(mesh_impl.ndims)
    group_info = groups_generator()
    for mesh_axis, (mesh_dim, tensor_axis) in enumerate(zip(
        mesh_impl.shape, tensor_axes)):
        slice_size = len(ret) // mesh_dim.size
        group_id, devices = next(group_info)
        if tensor_axis is None:
            start = group_id*slice_size
            ret = ret[start:start+slice_size]
            assert len(ret) == slice_size
        else:
            concat_inputs = []
            for i in range(slice_size):
                concat_inputs.append(
                        [ret[i + slice_size * j] for j in range(mesh_dim.size)])
            ret = mtf.parallel(
                    devices, tf.concat, concat_inputs,
                    axis=[tensor_axis] * len(devices))

    assert len(ret) == 1
    return ret[0]

# Replace mesh when a tensor is split along different axes in old and new
# meshes. Old and new meshes can have different sizes
class ReplaceMeshWithIndependentAxesOperation(MeshReplacementOperation):
    def lower(self, lowering):
        x = self.inputs[0]
        input_laid_out_tensor = lowering.tensors[x].to_laid_out_tensor()
        input_slices = input_laid_out_tensor.tensor_list

        old_mesh_impl = lowering.mesh_impl(self.old_mesh)
        new_mesh_impl = lowering.mesh_impl(self)
        old_mesh_shape = old_mesh_impl.shape
        new_mesh_shape = new_mesh_impl.shape
        old_mesh_size = old_mesh_shape.size
        new_mesh_size = new_mesh_shape.size

        assert ((old_mesh_impl.devices == new_mesh_impl.devices[:old_mesh_size]
                    and new_mesh_size % old_mesh_size == 0)
                or (old_mesh_impl.devices[:new_mesh_size] == new_mesh_impl.devices
                    and old_mesh_size % new_mesh_size == 0))

        # Remove/replicate redundant slices if needed
        def remove_replicate(slices):
            if new_mesh_size < old_mesh_size:
                slices = slices[:new_mesh_size]
            elif new_mesh_size > old_mesh_size:
                slices = (slices * (new_mesh_size // old_mesh_size))
            return slices

        # Get the set of axes to be split/concatenated
        split_axes, concat_axes = [], []
        for i, (old_dim, new_dim) in enumerate(zip(x.shape, self.new_shape)):
            old_axis = old_mesh_impl.tensor_dimension_to_mesh_axis(old_dim)
            new_axis = new_mesh_impl.tensor_dimension_to_mesh_axis(new_dim)

            is_old_dim_split = ((old_axis is not None) and
                    (old_mesh_shape[old_axis].size != 1))
            is_new_dim_split = ((new_axis is not None) and
                    (new_mesh_shape[new_axis].size != 1))
            if is_old_dim_split and is_new_dim_split:
                raise ValueError('Parallel axes for the tensor '
                        'in the old and new meshes should be independent.')

            if is_old_dim_split and (not is_new_dim_split):
                concat_axes.append((i, old_axis))
            elif (not is_old_dim_split) and is_new_dim_split:
                split_axes.append((i, new_axis))

        # Concatenate/split slices
        if not concat_axes:
            output_laid_out_tensor = new_mesh_impl.LaidOutTensor(
                    remove_replicate(input_slices))
            for ta, ma in split_axes:
                output_laid_out_tensor = new_mesh_impl.allsplit(
                        output_laid_out_tensor, ma, ta)

        elif not split_axes:
            output_laid_out_tensor = input_laid_out_tensor
            for ta, ma in concat_axes:
                output_laid_out_tensor = old_mesh_impl.allconcat(
                        output_laid_out_tensor, ma, ta)
            output_laid_out_tensor = new_mesh_impl.LaidOutTensor(
                    remove_replicate(output_laid_out_tensor.tensor_list))

        else:
            # Split slices along new axes
            slice_shape = mtf.Shape([mtf.Dimension(name, size) for name, size in
                zip(self.new_dim_names, input_slices[0].shape.as_list())])
            split_slices = [new_mesh_impl.make_slices(slice, slice_shape)
                    for slice in input_slices]
            split_slices = utils.TransposeLists(split_slices)
            assert len(split_slices) == new_mesh_size

            # Concat slices along old axes
            output_slices = [combine_slices(slice, x.shape, old_mesh_impl, i)
                    for i, slice in enumerate(split_slices)]
            assert len(output_slices) == new_mesh_shape.size
            output_laid_out_tensor = new_mesh_impl.LaidOutTensor(output_slices)

        lowering.set_tensor_lowering(self.outputs[0], output_laid_out_tensor)

def ReplaceMeshWithIndependentAxes(x, mesh, dim_names=None, name=None):
    return ReplaceMeshWithIndependentAxesOperation(x, mesh, dim_names,
            name=name).outputs[0]


# Split/concat slices along a mesh axis. We only consider tensors distributed
# along most-significant axis on both old and new meshes for now.
class ReplaceMeshWithConcatSplitOperation(MeshReplacementOperation):
    def lower(self, lowering):
        x = self.inputs[0]
        input_laid_out_tensor = lowering.tensors[x].to_laid_out_tensor()
        input_slices = input_laid_out_tensor.tensor_list

        old_mesh_impl = lowering.mesh_impl(self.old_mesh)
        new_mesh_impl = lowering.mesh_impl(self)
        old_mesh_shape = old_mesh_impl.shape
        new_mesh_shape = new_mesh_impl.shape
        old_mesh_ndims = old_mesh_shape.ndims
        new_mesh_ndims = new_mesh_shape.ndims
        if old_mesh_impl.devices != new_mesh_impl.devices:
            raise NotImplementedError

        old_ta2ma = old_mesh_impl.tensor_layout(
                x.shape).tensor_axis_to_mesh_axis
        new_ta2ma = new_mesh_impl.tensor_layout(
                self.new_shape).tensor_axis_to_mesh_axis

        # Get mesh and tensor axes along which to concat/split. We only allow
        # concat/split along one axis
        [(old_ta, old_ma)] = [(i, ma) for i, ma in enumerate(old_ta2ma)
                if ma is not None]
        [(new_ta, new_ma)] = [(i, ma) for i, ma in enumerate(new_ta2ma)
                if ma is not None]
        if old_ta != new_ta:
            raise NotImplementedError
        if not (old_ma == new_ma == 0):
            raise NotImplementedError

        # Perform concat/split
        old_axis_size = old_mesh_shape[old_ma].size
        new_axis_size = new_mesh_shape[new_ma].size
        if old_axis_size > new_axis_size:
            if new_mesh_ndims != 2:
                raise NotImplementedError
            output_laid_out_tensor = new_mesh_impl.allconcat(
                    input_laid_out_tensor, 1, new_ta)
        elif old_axis_size < new_axis_size:
            if old_mesh_ndims != 2:
                raise NotImplementedError
            output_laid_out_tensor = old_mesh_impl.allsplit(
                    input_laid_out_tensor, 1, old_ta)
        else:
            output_laid_out_tensor = input_laid_out_tensor

        lowering.set_tensor_lowering(self.outputs[0], output_laid_out_tensor)

def ReplaceMeshWithConcatSplit(x, mesh, dim_names=None, name=None):
    return ReplaceMeshWithConcatSplitOperation(x, mesh, dim_names,
            name=name).outputs[0]
