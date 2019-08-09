from operator import mul
from functools import reduce

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

from utils import TransposeLists, FlattenList, DeviceIndex


class ReplaceMeshOperation(mtf.Operation):
    def __init__(self, x, mesh, dim_names=None, name=None):
        self.old_mesh = x.mesh
        if isinstance(dim_names, mtf.Shape):
            dim_names = dim_names.dimension_names
        self.new_dim_names = dim_names = dim_names or x.shape.dimension_names
        assert x.mesh != mesh
        assert len(dim_names) == len(x.shape)
        self.new_shape = mtf.Shape([mtf.Dimension(name or dim.name, dim.size)
            for name, dim in zip(dim_names, x.shape.dims)])
        super().__init__([x], mesh=mesh, name=name or
                'replace_mesh')
        self._outputs = [mtf.Tensor(self, self.new_shape, x.dtype)]

    def gradient(self, grad_ys):
        return ReplaceMeshOperation(grad_ys[0], self.old_mesh,
                self.inputs[0].shape.dimension_names).outputs

    def lower(self, lowering):
        x = self.inputs[0]
        old_mesh_impl = lowering.mesh_impl(self.old_mesh)
        new_mesh_impl = lowering.mesh_impl(self)

        assert old_mesh_impl.shape.size == new_mesh_impl.shape.size

        # Make sure the slice shape is same in old and new mesh
        assert old_mesh_impl.slice_shape(x.shape)  \
                == new_mesh_impl.slice_shape(self.new_shape)

        # Make sure that each processor in old and new meshes have same slices
        # of the original tensor
        assert all(old_mesh_impl.slice_begin(x.shape, i) \
                == new_mesh_impl.slice_begin(self.new_shape, i) \
                for i in range(old_mesh_impl.shape.size))

        input_slices = lowering.tensors[x].to_laid_out_tensor().tensor_list
        laid_out_tensor = \
                new_mesh_impl.LaidOutTensor.from_tensor_list(input_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def ReplaceMesh(x, mesh, dim_names=None, name=None):
    return ReplaceMeshOperation(x, mesh, dim_names, name).outputs[0]


class ReplaceMeshWithDuplicatesOperation(mtf.Operation):
    def __init__(self, x, mesh, dim_names=None, name=None):
        self.old_mesh = x.mesh
        if isinstance(dim_names, mtf.Shape):
            dim_names = dim_names.dimension_names
        self.new_dim_names = dim_names = dim_names or x.shape.dimension_names
        assert x.mesh != mesh
        assert len(dim_names) == len(x.shape)
        self.new_shape = mtf.Shape([mtf.Dimension(name or dim.name, dim.size)
            for name, dim in zip(dim_names, x.shape.dims)])
        super().__init__([x], mesh=mesh, name=name or
                'replace_mesh_with_duplicates')
        self._outputs = [mtf.Tensor(self, self.new_shape, x.dtype)]

    def gradient(self, grad_ys):
        return ReplaceMeshWithDuplicatesOperation(grad_ys[0], self.old_mesh,
                self.inputs[0].shape.dimension_names).outputs

    def lower(self, lowering):
        x = self.inputs[0]
        input_slices = lowering.tensors[x].to_laid_out_tensor().tensor_list
        old_mesh_impl = lowering.mesh_impl(self.old_mesh)
        new_mesh_impl = lowering.mesh_impl(self)
        old_mesh_shape = old_mesh_impl.shape
        new_mesh_shape = new_mesh_impl.shape
        old_mesh_ndims = old_mesh_shape.ndims
        new_mesh_ndims = new_mesh_shape.ndims
        old_num_gpus = old_mesh_impl.shape.size
        new_num_gpus = new_mesh_impl.shape.size

        # Check if we need to replicate or remove slices
        replicate = (new_num_gpus > old_num_gpus)

        # Find the new-to-old pnum for devices of new mesh
        old_gpus = [DeviceIndex(d) for d in old_mesh_impl.devices]
        new_gpus = [DeviceIndex(d) for d in new_mesh_impl.devices]
        new_to_old_pnum = []
        for gpu in new_gpus:
            try:
                new_to_old_pnum.append(old_gpus.index(gpu))
            except ValueError:
                new_to_old_pnum.append(None)

        # If we need to replicate, find which pnums of old mesh to be used to
        # copy slices
        if replicate:
            new_ma2ta = new_mesh_impl.tensor_layout(
                    self.new_shape).mesh_axis_to_tensor_axis(new_mesh_ndims)
            axes = [i for i, ta in enumerate(new_ma2ta) if ta is None]
            pg = mtf.processor_groups(new_mesh_shape, axes)
            assert set(FlattenList(pg)) == set(range(new_num_gpus))

            for i, pnum in enumerate(new_to_old_pnum):
                if pnum is None:
                    for g in pg:
                        if i in g:
                            gpu = g[0]
                            break
                    new_to_old_pnum[i] = old_gpus.index(gpu)
        assert all(pnum is not None for pnum in new_to_old_pnum)

        # Make sure the slice shape is same in old and new mesh
        assert old_mesh_impl.slice_shape(x.shape)  \
                == new_mesh_impl.slice_shape(self.new_shape)

        # Make sure that each processor in old and new meshes have same slices
        # of the original tensor
        assert all(old_mesh_impl.slice_begin(x.shape, p_old) \
                == new_mesh_impl.slice_begin(self.new_shape, p_new) \
                for p_new, p_old in enumerate(new_to_old_pnum))

        output_slices = [input_slices[pnum] for pnum in new_to_old_pnum]
        laid_out_tensor = \
                new_mesh_impl.LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def ReplaceMeshWithDuplicates(x, mesh, dim_names=None, name=None):
    return ReplaceMeshWithDuplicatesOperation(x, mesh, dim_names,
            name).outputs[0]


class ReplaceMeshWithIndependentAxesOperation(mtf.Operation):
    # mesh: New mesh; dim_names: Dim names for 'x' in 'mesh'. If a dim_name is
    # None, current name of that axis is used.
    def __init__(self, x, mesh, dim_names=None, name=None):
        if isinstance(dim_names, mtf.Shape):
            dim_names = dim_names.dimension_names
        self.new_dim_names = dim_names = dim_names or x.shape.dimension_names
        assert x.mesh != mesh
        assert len(dim_names) == len(x.shape)
        self.old_mesh = x.mesh
        self.new_shape = mtf.Shape([mtf.Dimension(name or dim.name, dim.size)
            for name, dim in zip(dim_names, x.shape.dims)])
        super().__init__([x], mesh=mesh, name=name or
                'replace_mesh_independent_parallel_axes')
        self._outputs = [mtf.Tensor(self, self.new_shape, x.dtype)]

    def gradient(self, grad_ys):
        return ReplaceMeshWithIndependentAxesOperation(grad_ys[0],
                self.old_mesh, self.inputs[0].shape.dimension_names).outputs

    def lower(self, lowering):
        x = self.inputs[0]
        input_slices = lowering.tensors[x].to_laid_out_tensor().tensor_list
        old_mesh_impl = lowering.mesh_impl(self.old_mesh)
        new_mesh_impl = lowering.mesh_impl(self)
        new_mesh_shape = new_mesh_impl.shape

        for old_dim, new_dim in zip(x.shape, self.new_shape):
            old_axis = old_mesh_impl.tensor_dimension_to_mesh_axis(old_dim)
            new_axis = new_mesh_impl.tensor_dimension_to_mesh_axis(new_dim)
            if old_axis is not None and new_axis is not None:
                raise ValueError('Parallel axes for the tensor in the old and new \
                        meshes should be independent.')

        '''
        def GetParallelDims(mesh_impl, dims):
            axes = []
            for i, d in enumerate(dims):
                axis = mesh_impl.tensor_dimension_to_mesh_axis(d)
                if axis is not None:
                    axes.append(i)
            return axes

        # Find the tensor dimensions along which to concat, and split.
        # We need to concatenate along the axes that were split in old_mesh.
        # We need to split along new axes.
        concat_axes = GetParallelDims(old_mesh_impl, x.shape.dims)
        split_axes = GetParallelDims(new_mesh_impl, self.new_shape)
        if not (set(concat_axes) & set(split_axes)):
            raise ValueError('Parallel axes for the tensor in the old and new \
                    meshes should be independent.')

        # Split along new axes
        def Split(t, name='split'):
            ma2ta = new_mesh_impl.tensor_layout(self.new_shape) \
                    .mesh_axis_to_tensor_axis(new_mesh_shape.ndims)
            split_slices = [t]
            for i, ta, num_splits in enumerate(zip(ma2ta, new_mesh_shape)):
                if ta is None:
                    split_slices *= num_splits
                else:
                    tmp_slices = []
                    for slice in split_slices:
                        with tf.device(slice.device):
                            tmp_slices.append(tf.split(slice, num_splits,
                                axis=i, name={name}_{i}))
                    split_slices = FlattenList(TransposeLists(tmp_slices))

            assert len(split_slices) == new_mesh_shape.size
            return split_slices
        split_slices = [Split(slice) for slice in input_slices]
        '''

        # Split along new axes
        slice_shape = mtf.Shape([mtf.Dimension(name, size) for name, size in
            zip(self.new_dim_names, input_slices[0].get_shape().as_list())])
        split_slices = [new_mesh_impl.make_slices(slice, slice_shape) \
                for slice in input_slices]
        assert all(len(slice) == new_mesh_shape.size for slice in split_slices)

        # Concat along old axes
        output_slices = []
        for dev, slices in zip(new_mesh_impl.devices, zip(*split_slices)):
            output_slices.append(old_mesh_impl.combine_slices(slices,
                x.shape, device=dev))
        assert len(output_slices) == new_mesh_shape.size

        laid_out_tensor = lowering.mesh_impl(
                self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def ReplaceMeshWithIndependentAxes(x, mesh, dim_names=None, name=None):
    return ReplaceMeshWithIndependentAxesOperation(x, mesh, dim_names,
            name=name).outputs[0]


class ReplaceMeshWithConcatSplitOperation(mtf.Operation):
    def __init__(self, x, mesh, dim_names=None, name=None):
        self.old_mesh = x.mesh
        if isinstance(dim_names, mtf.Shape):
            dim_names = dim_names.dimension_names
        self.new_dim_names = dim_names = dim_names or x.shape.dimension_names
        assert x.mesh != mesh
        assert len(dim_names) == len(x.shape)
        self.new_shape = mtf.Shape([mtf.Dimension(name or dim.name, dim.size)
            for name, dim in zip(dim_names, x.shape.dims)])
        super().__init__([x], mesh=mesh, name=name or
                'replace_mesh_with_duplicates')
        self._outputs = [mtf.Tensor(self, self.new_shape, x.dtype)]

    def gradient(self, grad_ys):
        return ReplaceMeshWithConcatSplitOperation(grad_ys[0],
                self.old_mesh, self.inputs[0].shape.dimension_names).outputs

    def lower(self, lowering):
        x = self.inputs[0]
        input_slices = lowering.tensors[x].to_laid_out_tensor().tensor_list
        old_mesh_impl = lowering.mesh_impl(self.old_mesh)
        new_mesh_impl = lowering.mesh_impl(self)
        old_mesh_shape = old_mesh_impl.shape
        new_mesh_shape = new_mesh_impl.shape
        old_mesh_ndims = old_mesh_shape.ndims
        new_mesh_ndims = new_mesh_shape.ndims
        old_num_gpus = old_mesh_impl.shape.size
        new_num_gpus = new_mesh_impl.shape.size
        assert old_mesh_ndims == new_mesh_ndims

        old_ta2ma = old_mesh_impl.tensor_layout(
                x.shape).tensor_axis_to_mesh_axis
        new_ta2ma = new_mesh_impl.tensor_layout(
                self.new_shape).tensor_axis_to_mesh_axis
        assert old_ta2ma == new_ta2ma

        concat_mesh_dim = -1
        concat_tensor_dim = -1
        concat = False
        for i, dim in enumerate(old_ta2ma):
            if dim == None:
                continue

            size1 = old_mesh_shape[dim].size
            size2 = new_mesh_shape[dim].size
            if size1 == size2:
                continue

            assert concat_mesh_dim == -1
            concat_tensor_dim = i
            concat_mesh_dim = dim
            if size1 > size2:
                concat = True

        assert concat_mesh_dim != -1
        old_groups = mtf.processor_groups(old_mesh_shape, [concat_mesh_dim])
        new_groups = mtf.processor_groups(new_mesh_shape, [concat_mesh_dim])
        assert len(old_groups) == len(new_groups)
        assert all(len(old_groups[0]) == len(group) for group in old_groups)
        assert all(len(new_groups[0]) == len(group) for group in new_groups)
        old_gpus = [DeviceIndex(d) for d in old_mesh_impl.devices]
        new_gpus = [DeviceIndex(d) for d in new_mesh_impl.devices]

        if concat:
            assert len(old_groups[0]) % len(new_groups[0]) == 0
            num_splits = len(old_groups[0]) // len(new_groups[0])
        else:
            assert len(new_groups[0]) % len(old_groups[0]) == 0
            num_splits = len(new_groups[0]) // len(old_groups[0])

        output_slices = [None] * new_num_gpus
        for old_group, new_group in zip(old_groups, new_groups):
            if concat:
                for idx1, idx2 in enumerate(range(0, len(old_group),
                    num_splits)):
                    slices = [input_slices[p] for p in
                            old_group[idx2:idx2+num_splits]]
                    assert output_slices[new_group[idx1]] == None
                    assert old_gpus[old_group[idx2]] == new_gpus[new_group[idx1]]

                    with tf.device(new_mesh_impl.devices[new_group[idx1]]):
                        output_slices[new_group[idx1]] = tf.concat(slices,
                                axis=concat_tensor_dim)

            else:
                for idx1, idx2 in enumerate(range(0, len(new_group),
                    num_splits)):
                    slice = input_slices[old_group[idx1]]
                    assert all(output_slices[p] == None for p in
                            new_group[idx2:idx2+num_splits])
                    assert old_gpus[old_group[idx1]] == new_gpus[new_group[idx2]]

                    with tf.device(new_mesh_impl.devices[new_group[idx2]]):
                        slices = tf.split(slice, num_splits,
                                axis=concat_tensor_dim)

                    for p, slice in zip(new_group[idx2:idx2+num_splits], slices):
                        with tf.device(new_mesh_impl.devices[p]):
                            assert output_slices[p] == None
                            output_slices[p] = tf.identity(slice)

        assert all(s is not None for s in output_slices)
        laid_out_tensor = lowering.mesh_impl(
                self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def ReplaceMeshWithConcatSplit(x, mesh, dim_names=None, name=None):
    return ReplaceMeshWithConcatSplitOperation(x, mesh, dim_names,
            name=name).outputs[0]


