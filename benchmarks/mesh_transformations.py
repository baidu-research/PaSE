from operator import mul
from functools import reduce

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

from utils import TransposeLists, FlattenList


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


def ReplaceMeshWithReplication(new_mesh, tsr, axis, name=None):
    return ReplaceMeshWithReplicationOperation(new_mesh, tsr, axis,
            name).outputs[0]


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


def ReplaceMeshWithRemoval(new_mesh, tsr, axis, name=None):
    return ReplaceMeshWithRemovalOperation(new_mesh, tsr, axis, name).outputs[0]


class ReplaceMeshWithIndependentAxesOperation(mtf.Operation):
    # mesh: New mesh; dim_names: Dim names for 'x' in 'mesh'
    def __init__(self, x, mesh, dim_names, name=None):
        assert len(dim_names) == len(x.shape)
        self.old_mesh = x.mesh
        self.new_dim_names = dim_names
        self.new_shape = mtf.Shape([mtf.Dimension(name, dim.size) \
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

