from operator import mul
from functools import reduce

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

from utils import TransposeLists, FlattenList


class ReplaceMeshWithDuplicatesOperation(mtf.Operation):
    def __init__(self, x, mesh, dim_names, name=None):
        assert x.mesh != mesh
        assert len(dim_names) == len(x.shape)
        self.old_mesh = x.mesh
        self.new_dim_names = dim_names
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
        old_gpus = [d.device_index for d in old_mesh_impl.devices]
        new_gpus = [d.device_index for d in new_mesh_impl.devices]
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


def ReplaceMeshWithDuplicates(x, mesh, dim_names, name=None):
    return ReplaceMeshWithDuplicatesOperation(x, mesh, dim_names, name).outputs[0]


class ReplaceMeshWithIndependentAxesOperation(mtf.Operation):
    # mesh: New mesh; dim_names: Dim names for 'x' in 'mesh'. If a dim_name is
    # None, current name of that axis is used.
    def __init__(self, x, mesh, dim_names, name=None):
        assert x.mesh != mesh
        assert len(dim_names) == len(x.shape)
        self.old_mesh = x.mesh
        self.new_dim_names = dim_names
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

        laid_out_tensor = \
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def ReplaceMeshWithIndependentAxes(x, mesh, dim_names, name=None):
    return ReplaceMeshWithIndependentAxesOperation(x, mesh, dim_names,
            name=name).outputs[0]

