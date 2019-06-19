from operator import mul
from functools import reduce

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf


def Prod(lst):
    return reduce(mul, lst, 1)


def GetDeviceList(num_gpus):
    return ['gpu:%d' %i for i in range(num_gpus)]


def AssignLayout(ta_axes, mesh_axis):
    layout = []
    for a in ta_axes:
        a = a.name if isinstance(a, mtf.Dimension) else a
        layout.append((a, mesh_axis))
    return layout


def GetMeshImpl(devices, layout=None):
    layout = layout or [['ma%d' % i] for i in range(len(devices))]
    assert len(devices) == len(layout)

    mesh_shape = []
    layout_rules = []
    for i, (d, ls) in enumerate(zip(devices, layout)):
        p_name = 'p%d' % i
        mesh_shape.append((p_name, d))
        for l in ls:
            layout_rules.append((p_name, l))

    devices = GetDeviceList(Prod(devices))
    return mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules,
            devices)


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


def NormalizeStrideAndPad(stride, padding):
    stride = MakePair(stride)

    if isinstance(padding, str):
        assert padding == 'VALID' or padding == 'SAME'
    else:
        if padding == 0:
            padding = 'VALID'
        else:
            padding = 'SAME'

    return stride, padding


# Fixes a bug in mesh-tensorflow. To be used as gradient function for slicewise
# op. Uncomment this part if needed later.
'''
class GenericGradOp(mtf.GenericGradOperation):
    def lower(self, lowering):
        # lists of lists of tf.Tensor
        all_ys = mtf.transpose_list_of_lists(
            [lowering.tensors[y].to_laid_out_tensor().tensor_list for y in self._forward_op.outputs])
        all_xs = mtf.transpose_list_of_lists(
            [lowering.tensors[x].to_laid_out_tensor().tensor_list for x in self._forward_op.inputs])
        all_grad_ys = mtf.transpose_list_of_lists(
            [lowering.tensors[dy].to_laid_out_tensor().tensor_list for dy in self._grad_ys])
        all_grad_xs = [tf.gradients(ys=ys, xs=xs, grad_ys=grad_ys) for
                       ys, xs, grad_ys in zip(all_ys, all_xs, all_grad_ys)]
        grad_xs = mtf.transpose_list_of_lists(all_grad_xs)
        for out, grad_x in zip(self.outputs, grad_xs):
            lowering.set_tensor_lowering( out,
                    lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(grad_x))


def GenericGradFn(forward_op, grad_y):
    return GenericGradOp(forward_op, [grad_y]).outputs
'''


class Conv2dOperation(mtf.Conv2dOperation):
    def __init__(self, conv_input, conv_filter, strides, padding, name=None):
        mtf.Operation.__init__(self, [conv_input, conv_filter], name=name or
                "conv2d")
        self._padding = padding
        self._batch_dims = conv_input.shape.dims[:-3]
        self._in_h_dim, self._in_w_dim, self._in_dim = conv_input.shape.dims[-3:]
        self._fh_dim, self._fw_dim = conv_filter.shape.dims[:2]
        f_in_dim, self._out_dim = conv_filter.shape.dims[2:]
        if f_in_dim != self._in_dim:
          raise ValueError("Dimensions do not match input=%s filter=%s"
                           % (conv_input, conv_filter))
        out_h = self._in_h_dim.size
        out_w = self._in_w_dim.size
        if padding == "VALID":
            out_h -= self._fh_dim.size
            out_w -= self._fw_dim.size

        self._strides = strides
        if strides is not None:
            out_h //= strides[1]
            out_w //= strides[2]

        if padding == "VALID":
            out_h += 1
            out_w += 1

        self._out_h_dim = mtf.Dimension(self._in_h_dim.name, out_h)
        self._out_w_dim = mtf.Dimension(self._in_w_dim.name, out_w)
        output_shape = mtf.Shape(
            self._batch_dims + [self._out_h_dim, self._out_w_dim, self._out_dim])
        self._outputs = [mtf.Tensor(self, output_shape, conv_input.dtype)]


def Conv2d(tsr, fltr_shape, stride=(1,1), padding='VALID', use_bias=True,
        activation=None, name=None):
    stride, padding = NormalizeStrideAndPad(stride, padding)
    with tf.variable_scope(name, default_name='conv2d'):
        assert tsr.shape[-1] == fltr_shape[-2]

        w = mtf.get_variable(tsr.mesh, 'weight', fltr_shape, dtype=tsr.dtype)
        out = Conv2dOperation(tsr, w, (1, stride[0], stride[1], 1),
                padding).outputs[0]

        if use_bias == True:
            b = mtf.get_variable(tsr.mesh, 'bias', mtf.Shape([out.shape[-1]]),
                    initializer=tf.zeros_initializer(), dtype=tsr.dtype)
            out += b

        if activation is not None:
            out = activation(out)

        return out


def Pooling(tsr, fltr, stride=(1,1), padding='VALID', pooling_fn=tf.nn.max_pool,
        name=None):
    stride, padding = NormalizeStrideAndPad(stride, padding)
    with tf.variable_scope(name, default_name='pool'):
        def max_pool(x):
            return pooling_fn(x, [1, fltr[0], fltr[1], 1], [1, stride[0],
                stride[1], 1], padding)

        # Output shape
        h_o = tsr.shape[1].size
        w_o = tsr.shape[2].size
        if padding == 'VALID':
            h_o -= fltr[0]
            w_o -= fltr[1]
        h_o //= stride[0]
        w_o //= stride[1]
        if padding == 'VALID':
            h_o += 1
            w_o += 1

        output_shape = tsr.shape.resize_dimension(tsr.shape[1].name, h_o)
        output_shape = output_shape.resize_dimension(tsr.shape[2].name, w_o)

        splittable_dims = [tsr.shape.dims[0], tsr.shape.dims[-1]]
        #out = mtf.slicewise(max_pool, [tsr], output_shape, tsr.dtype,
        #        splittable_dims, grad_function=GenericGradFn)
        out = mtf.slicewise(max_pool, [tsr], output_shape, tsr.dtype,
                splittable_dims)
        return out


def MaxPool(*args, **kwargs):
    return Pooling(*args, pooling_fn=tf.nn.max_pool, **kwargs)


def AvgPool(*args, **kwargs):
    return Pooling(*args, pooling_fn=tf.nn.avg_pool, **kwargs)


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


def ReplaceMeshWithRemoval(new_mesh, tsr, axis, name=None):
    return ReplaceMeshWithRemovalOperation(new_mesh, tsr, axis, name).outputs[0]

