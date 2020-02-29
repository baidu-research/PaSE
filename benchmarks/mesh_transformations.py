from operator import mul
from functools import reduce

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import utils
from utils import TransposeLists, FlattenList, DeviceIndex, RandName

'''
def HasDGXLink(x, y):
    return (x//4 == y//4) or (y == x+4)

def PnumToDeviceID(mesh_impl, pnums):
    return [DeviceIndex(mesh_impl.devices[p]) for p in pnums]

# Replication function specialized for DGX network topology
def ReplicateOnDGX(tsrs, gpu_ids):
    if isinstance(gpu_ids[0], str) or isinstance(gpu_ids[0], tf.DeviceSpec):
        gpu_ids = [DeviceIndex(g) for g in gpu_ids]

    num_gpus = len(gpu_ids)
    assert len(tsrs) <= 4
    assert bool(num_gpus and 
            not (num_gpus & (num_gpus-1))) # num_gpus is a power of 2

    gpu_ids = sorted(gpu_ids)
    assert gpu_ids[-1] < 8
    assert all((not tsr.device) or (
            DeviceIndex(tsr.device) == gpu_ids[i]) 
            for i, tsr in enumerate(tsrs))

    for i in range(len(tsrs)):
        with tf.device(f'/device:GPU:{gpu_ids[i]}'):
            tsrs[i] = tf.identity(tsrs[i])

    if len(tsrs) < 2:
        with tf.device(f'/device:GPU:{gpu_ids[1]}'):
            assert HasDGXLink(gpu_ids[0], gpu_ids[1])
            tsrs.append(tf.identity(tsrs[0]))

    if len(tsrs) < 4 and num_gpus > 2:
        for i in range(2):
            with tf.device(f'/device:GPU:{gpu_ids[i+2]}'):
                assert HasDGXLink(gpu_ids[i], gpu_ids[i+2])
                tsrs.append(tf.identity(tsrs[i]))

    if num_gpus > 4:
        for i in range(4):
            with tf.device(f'/device:GPU:{gpu_ids[i+4]}'):
                assert HasDGXLink(gpu_ids[i], gpu_ids[i+4])
                tsrs.append(tf.identity(tsrs[i]))

    assert len(tsrs) == num_gpus
    return tsrs
'''

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
    def __init__(self, x, mesh, dim_names=None, axis=-1, name=None):
        self.old_mesh = x.mesh
        self.axis = axis
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
                self.inputs[0].shape.dimension_names, axis=self.axis).outputs

    def lower(self, lowering):
        x = self.inputs[0]
        input_slices = lowering.tensors[x].to_laid_out_tensor().tensor_list
        old_mesh_impl = lowering.mesh_impl(self.old_mesh)
        new_mesh_impl = lowering.mesh_impl(self)
        old_mesh_shape = old_mesh_impl.shape
        new_mesh_shape = new_mesh_impl.shape
        old_num_gpus = old_mesh_impl.shape.size
        new_num_gpus = new_mesh_impl.shape.size
        replicate = (new_num_gpus > old_num_gpus)

        if new_num_gpus == old_num_gpus:
            assert old_mesh_impl.devices == new_mesh_impl.devices
            laid_out_tensor = \
                    new_mesh_impl.LaidOutTensor.from_tensor_list(input_slices)
            lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)
            return

        # Make sure the slice shape is same in old and new mesh
        assert old_mesh_impl.slice_shape(x.shape)  \
                == new_mesh_impl.slice_shape(self.new_shape)

        assert (old_mesh_shape.ndims == new_mesh_shape.ndims) or (self.axis >= 0
                and abs(old_mesh_shape.ndims - new_mesh_shape.ndims) == 1)
        if old_mesh_shape.ndims < new_mesh_shape.ndims:
            mesh_dims = old_mesh_shape.dims
            mesh_dims.insert(self.axis, mtf.Dimension(RandName(), 1))
            old_mesh_shape = mtf.Shape(mesh_dims)
        elif old_mesh_shape.ndims > new_mesh_shape.ndims:
            mesh_dims = new_mesh_shape.dims
            mesh_dims.insert(self.axis, mtf.Dimension(RandName(), 1))
            new_mesh_shape = mtf.Shape(mesh_dims)
        old_mesh_dims = old_mesh_shape.to_integer_list
        new_mesh_dims = new_mesh_shape.to_integer_list

        # Initialize output_slices with None
        output_slices = [None] * new_num_gpus
        def FillOutputSlices(slices, pnums):
            assert len(slices) == len(pnums)
            for p, tsr in zip(pnums, slices):
                assert output_slices[p] == None
                output_slices[p] = tsr

        # Fill output_slices
        new_to_old_pnum = [None] * new_num_gpus
        for i, (o, n) in enumerate(zip(old_mesh_dims, new_mesh_dims)):
            old_pg = mtf.processor_groups(old_mesh_shape, [i])
            new_pg = mtf.processor_groups(new_mesh_shape, [i])
            if replicate:
                assert n >= o and n % o == 0
                assert len(new_pg) >= len(old_pg)
            else:
                assert o >= n and o % n == 0
                assert len(old_pg) >= len(new_pg)

            if n == o:
                continue

            for old_g, new_g in zip(old_pg, new_pg):
                tsrs = [input_slices[p] for p in old_g]
                if n > o:
                    #tsrs = ReplicateOnDGX(tsrs, PnumToDeviceID(new_mesh_impl,
                    #    new_g))
                    assert utils.is_power_of_2(len(old_g))
                    assert utils.is_power_of_2(len(new_g))
                    assert all(old_mesh_impl.devices[d1] ==
                            new_mesh_impl.devices[d2] for d1, d2 in zip(old_g,
                                new_g))

                    while len(tsrs) < len(new_g):
                        new_tsrs = []
                        for t, g in zip(tsrs, new_g[len(tsrs):]):
                            with tf.device(new_mesh_impl.devices[g]):
                                new_tsrs.append(t)
                        tsrs += new_tsrs
                    assert len(tsrs) == len(new_g)
                else:
                    tsrs = tsrs[:n]

                if __debug__:
                    old_gs = old_g[:]
                    new_gs = new_g[:]
                    if len(old_gs) < len(new_gs):
                        while len(old_gs) < len(new_gs):
                            old_gs *= 2
                        assert len(old_gs) == len(new_gs)
                    for p_old, p_new in zip(old_gs, new_gs):
                        assert new_to_old_pnum[p_new] == None
                        new_to_old_pnum[p_new] = p_old
                FillOutputSlices(tsrs, new_g)

        # Make sure that each processor in old and new meshes have same slices
        # of the original tensor
        assert all(g is not None for g in new_to_old_pnum)
        assert all(old_mesh_impl.slice_begin(x.shape, p_old) \
                == new_mesh_impl.slice_begin(self.new_shape, p_new) \
                for p_new, p_old in enumerate(new_to_old_pnum))

        assert all(s is not None for s in output_slices)
        laid_out_tensor = \
                new_mesh_impl.LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def ReplaceMeshWithDuplicates(x, mesh, dim_names=None, axis=-1, name=None):
    return ReplaceMeshWithDuplicatesOperation(x, mesh, dim_names,
            axis, name).outputs[0]


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

        # Split along new axes
        slice_shape = mtf.Shape([mtf.Dimension(name, size) for name, size in
            zip(self.new_dim_names, input_slices[0].get_shape().as_list())])
        split_slices = [new_mesh_impl.make_slices(slice, slice_shape) \
                for slice in input_slices]
        assert all(len(s) == new_mesh_shape.size for s in split_slices)
        for s in split_slices:
            for i, d in enumerate(new_mesh_impl.devices):
                with tf.device(d):
                    s[i] = tf.identity(s[i])
        assert all(len(slice) == new_mesh_shape.size for slice in split_slices)

        # Concat along old axes
        output_slices = []
        for dev, slices in zip(new_mesh_impl.devices, zip(*split_slices)):
            assert all(s.device == dev for s in slices)
            output_slices.append(old_mesh_impl.combine_slices(slices,
                x.shape, device=dev))
        assert len(output_slices) == new_mesh_shape.size

        laid_out_tensor = lowering.mesh_impl(
                self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def ReplaceMeshWithIndependentAxes(x, mesh, dim_names=None, name=None):
    return ReplaceMeshWithIndependentAxesOperation(x, mesh, dim_names,
            name=name).outputs[0]


# Split/concat slices along a mesh axis. We only consider tensors distributed
# along one axis for now.
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
                'replace_mesh_with_concat_split')
        self._outputs = [mtf.Tensor(self, self.new_shape, x.dtype)]

    def gradient(self, grad_ys):
        return ReplaceMeshWithConcatSplitOperation(grad_ys[0], self.old_mesh,
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

        num_gpus = old_mesh_impl.shape.size
        assert num_gpus == new_mesh_impl.shape.size

        old_ta2ma = old_mesh_impl.tensor_layout(
                x.shape).tensor_axis_to_mesh_axis
        new_ta2ma = new_mesh_impl.tensor_layout(
                self.new_shape).tensor_axis_to_mesh_axis
        assert old_ta2ma == new_ta2ma

        axis = -1
        old_mesh_axis = -1
        new_mesh_axis = -1
        for i, (ma1, ma2) in enumerate(zip(old_ta2ma, new_ta2ma)):
            # Either both ma1 and ma2 are none, or neither is none
            if ma1 is None:
                assert ma2 is None
                continue
            assert ma2 is not None

            # We consider only the case where tensor is distributed along only
            # one axis.
            assert axis == old_mesh_axis == new_mesh_axis == -1
            old_mesh_axis, new_mesh_axis = ma1, ma2
            axis = i

        old_axis_size = old_mesh_shape[old_mesh_axis].size
        new_axis_size = new_mesh_shape[new_mesh_axis].size

        if old_axis_size == new_axis_size:
            # If old and new axis sizes are same, there is nothing to
            # concat/split.  Just return the original slices
            output_slices = input_slices

        elif old_axis_size > new_axis_size: # Concat
            # We only consider the case where there is no replication in old
            # mesh.
            assert old_axis_size == num_gpus

            assert old_axis_size % new_axis_size == 0
            ratio = old_axis_size // new_axis_size

            # We only consider this case for now, so we don't have to shuffle
            # the concatenated slices
            assert new_mesh_axis == 0

            # Concat 'ratio' slices together
            output_slices = []
            for i in range(0, old_axis_size, ratio):
                devices = [new_mesh_impl.devices[j] for j in range(i, i+ratio)]
                output_slices += mtf.placement_mesh_impl.allconcat_ring(
                        [input_slices[j] for j in range(i, i+ratio)],
                        devices, axis)

        else: # Split
            # We only consider the case where there is no replication in new
            # mesh.
            assert new_axis_size == num_gpus

            assert new_axis_size % old_axis_size == 0
            ratio = new_axis_size // old_axis_size

            output_slices = []
            t_size = input_slices[0].shape[axis]
            for i in range(0, new_axis_size, ratio):
                start = 0
                assert t_size % ratio == 0
                step = t_size // ratio
                slices = [slice(None) for _ in x.shape.dims]
                slices[axis] = slice(start, start+step)

                for j in range(i, i+ratio):
                    t = input_slices[j]
                    assert t.device == new_mesh_impl.devices[j]

                    with tf.device(t.device):
                        output_slices.append(t[slices])
                    start += step
                assert start == t_size

        laid_out_tensor = lowering.mesh_impl(
                self).LaidOutTensor.from_tensor_list(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)


def ReplaceMeshWithConcatSplit(x, mesh, dim_names=None, name=None):
    return ReplaceMeshWithConcatSplitOperation(x, mesh, dim_names,
            name=name).outputs[0]


