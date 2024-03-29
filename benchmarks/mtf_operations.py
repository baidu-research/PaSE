import numpy as np
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf 

import utils

# Fixes a bug in mesh-tensorflow. To be used as gradient function for slicewise op.
# Also, mtf.GenericGradOperation does not colocate_gradients.
class GenericGradOperation(mtf.GenericGradOperation):
    def lower(self, lowering):
        # lists of lists of tf.Tensor
        all_ys = mtf.transpose_list_of_lists(
            [lowering.tensors[y].to_laid_out_tensor().tensor_list for y in self._forward_op.outputs])
        all_xs = mtf.transpose_list_of_lists(
            [lowering.tensors[x].to_laid_out_tensor().tensor_list for x in self._forward_op.inputs])
        all_grad_ys = mtf.transpose_list_of_lists(
            [lowering.tensors[dy].to_laid_out_tensor().tensor_list for dy in self._grad_ys])
        all_grad_xs = [tf.gradients(ys=ys, xs=xs, grad_ys=grad_ys,
            colocate_gradients_with_ops=True)
                for ys, xs, grad_ys in zip(all_ys, all_xs, all_grad_ys)]
        grad_xs = mtf.transpose_list_of_lists(all_grad_xs)
        for out, grad_x in zip(self.outputs, grad_xs):
            lowering.set_tensor_lowering(out,
                    lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(grad_x))

def GenericGradFn(forward_op, grad_y):
    return GenericGradOperation(forward_op, [grad_y]).outputs


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

def NormalizeStrideAndPad(stride, padding):
    stride = utils.MakePair(stride)

    if isinstance(padding, str):
        assert padding == 'VALID' or padding == 'SAME'
    else:
        if padding == 0:
            padding = 'VALID'
        else:
            padding = 'SAME'

    return stride, padding

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
        out = mtf.slicewise(max_pool, [tsr], output_shape, tsr.dtype,
                splittable_dims, grad_function=GenericGradFn)
        return out

def MaxPool(*args, **kwargs):
    return Pooling(*args, pooling_fn=tf.nn.max_pool, **kwargs)

def AvgPool(*args, **kwargs):
    return Pooling(*args, pooling_fn=tf.nn.avg_pool, **kwargs)


# Mesh-tensorflow's reshape operation produces wrong results when a dim of a
# tensor is split along one mesh axis, and it is renamed to be split along a
# different mesh axis. So, we first combine the slices along old mesh axis, and
# split along new mesh axis
def rename_dimension(x, old_dim_name, new_dim_name):
    assert isinstance(x, mtf.Tensor)
    if old_dim_name == new_dim_name:
        return x

    if old_dim_name.startswith('axis') and new_dim_name.startswith('axis'):
        tmp_dim_name = utils.RandName()
        x = mtf.rename_dimension(x, old_dim_name, tmp_dim_name)
        old_dim_name = tmp_dim_name

    return mtf.rename_dimension(x, old_dim_name, new_dim_name)
RenameDimension = rename_dimension

# Mesh-tensorflow's reshape operation produces wrong results when a dim of a
# tensor is split along one mesh axis, and it is renamed to be split along a
# different mesh axis. So, we first combine the slices along old mesh axis, and
# split along new mesh axis
def reshape(x, new_shape):
    old_shape = x.shape
    assert len(old_shape) == len(new_shape)
    for o, n in zip(old_shape.dims, new_shape.dims):
        if (o.name != n.name) and (o.name.startswith('axis') and
                n.name.startswith('axis')):
            x = mtf.rename_dimension(x, o.name, utils.RandName())
    return mtf.reshape(x, new_shape)
