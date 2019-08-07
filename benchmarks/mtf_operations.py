import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf 

import utils

# Allows concatenation of distributed dimensions. When the concat_axis is
# distributed, slices in each processor are locally concatenated. No
# communication happens.
class ConcatOperation(mtf.ConcatOperation):
    def gradient(self, grad_ys):
        dy = grad_ys[0]
        return split(dy, self.outputs[0].shape.dims[self._axis],
                self._input_sizes)

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        def slicewise_fn(*args):
            return tf.concat(args, axis=self._axis, name="concat")
        y = mesh_impl.slicewise(
                slicewise_fn, *[lowering.tensors[x] for x in self._inputs])
        lowering.set_tensor_lowering(self.outputs[0], y)


# Allows splitting of distributed dimensions. When the split_axis is
# distributed, slices in each processor are locally split. No
# communication happens.
class SplitOperation(mtf.SplitOperation):
    def gradient(self, grad_ys):
        return [concat(grad_ys, self._split_dim.name)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        ma = mesh_impl.tensor_dimension_to_mesh_axis(self._split_dim)
        if ma is not None:
            axis_size = mesh_impl.shape[ma].size
            output_sizes = [s // axis_size for s in self._output_sizes]
        else:
            output_sizes = self._output_sizes
        def slicewise_fn(x):
            return tuple(tf.split(x, output_sizes, axis=self._axis))
        values = mesh_impl.slicewise(
                slicewise_fn, lowering.tensors[self.inputs[0]])
        for t, v in zip(self._outputs, values):
            lowering.set_tensor_lowering(t, v)


def concat(xs, concat_dim_name, name=None):
    return ConcatOperation(xs, concat_dim_name, name).outputs[0] 


def split(x, split_dim, num_or_size_splits, name=None):
    return SplitOperation(x, split_dim, num_or_size_splits, name=name).outputs


# Extends mesh-tensorflow's WhileLoopOperation with gradient, but only accepts
# a simple loop counter that starts at 0 and increments by 1 at each iteration,
# until it reaches an upper-bound.
class WhileLoopWithGradOperation(mtf.WhileLoopOperation):
    def __init__(self, cond_fn, body_fn, cond_ub, inputs, tf_kwargs=None,
            name="while_loop"):
        self.cond_ub = cond_ub
        super().__init__(cond_fn, body_fn, inputs, tf_kwargs, name)

    def gradient(self, grad_ys):
        op_body_inputs = self._body_inputs
        op_body_outputs = self._body_outputs

        def grad_loop_fn(*fn_inputs):
            downstream = set(op_body_inputs)
            for op in self._body_ops:
                if op.has_gradient:
                    if set(op.inputs) & downstream:
                        downstream |= set(op.outputs)
            tensor_to_gradient = dict(zip(op_body_outputs, fn_inputs))
            #for out, g, inp in zip(op_body_outputs, grad_ys, fn_inputs):
            #    if g is not None:
            #        tensor_to_gradient[out] = inp
            for op in self._body_ops[::-1]:
                grad_outputs = [tensor_to_gradient.get(out, mtf.zeros_like(out))
                        for out in op.outputs]
                if op.has_gradient and any(grad_outputs) and (set(op.inputs) & downstream):
                    with tf.variable_scope(op.name + "/gradients"):
                        print(op.gradient, grad_outputs)
                        input_grads = op.gradient(grad_outputs)
                        for inp, grad in zip(op.inputs, input_grads):
                            if inp in downstream and grad is not None:
                                if inp in tensor_to_gradient:
                                    tensor_to_gradient[inp] += grad
                                else:
                                    tensor_to_gradient[inp] = grad
            fn_outputs = []
            for i, x in enumerate(op_body_inputs):
                try:
                    grad = tensor_to_gradient[x]
                    fn_outputs.append(grad)
                except KeyError:
                    grad = mtf.zeros_like(x)
                    fn_outputs.append(grad)
                    tensor_to_gradient[x] = grad
            return fn_outputs

        xs = [mtf.zeros_like(self.outputs[i]) if x is None else x for i, x in
                enumerate(grad_ys)]
        ys = WhileLoop(self.cond_ub, grad_loop_fn, xs, name='while_loop_gradient_loop')
        return ys

def WhileLoop(cond_ub, body_fn, inputs, name='while_loop'):
    assert isinstance(cond_ub, int)
    init = mtf.constant(inputs[0].mesh, 0, dtype=tf.int32)
    ub = mtf.constant(inputs[0].mesh, cond_ub, dtype=tf.int32)

    def my_cond_fn(*kwargs):
        return mtf.less(kwargs[0], ub)

    def my_body_fn(*kwargs):
        inc = mtf.add(kwargs[0], 1)
        outputs = tuple(body_fn(*kwargs[1:]))
        return (inc,) + outputs

    my_inputs = (init,) + tuple(inputs)
    return WhileLoopWithGradOperation(my_cond_fn, my_body_fn, cond_ub, my_inputs,
            name=name).outputs[1:]


## Similar to tf.Tensor.__getitem__
#class GetItemOperation(mtf.SliceOperation):
#    def lower(self, lowering):
#        mesh_impl = lowering.mesh_impl(self)
#        if mesh_impl.tensor_dimension_to_mesh_axis(self._slice_dim) is not None:
#            raise ValueError("can't slice along split axis")
#        inputs = self._inputs[0]
#        ndims = self._inputs[0].shape.ndims
#        axis = self._axis
#        slices = [slice(0, -1)] * axis \
#                + [slice(self._begin, self._begin + self._slice_dim.size)] \
#                + [slice(0, -1)] * (ndims - axis - 1)
#
#        def slicewise_fn(x, begin, size):
#            return x.__getitem__(slices)
#        y = mesh_impl.slicewise(
#            slicewise_fn, lowering.tensors[inputs], begin, size)
#        lowering.set_tensor_lowering(self.outputs[0], y)
#
#def GetItem(x, begin, size, slice_dim_name, name=None):
#    return GetItemOperation(x, begin, size, slice_dim_name,
#            name=name).outputs[0]

# Fixes a bug in mesh-tensorflow. To be used as gradient function for slicewise op.
# Uncomment this part if not needed later.
class GenericGradOp(mtf.GenericGradOperation):
    def lower(self, lowering):
        # lists of lists of tf.Tensor
        all_ys = mtf.transpose_list_of_lists(
            [lowering.tensors[y].to_laid_out_tensor().tensor_list for y in self._forward_op.outputs])
        all_xs = mtf.transpose_list_of_lists(
            [lowering.tensors[x].to_laid_out_tensor().tensor_list for x in self._forward_op.inputs])
        all_grad_ys = mtf.transpose_list_of_lists(
            [lowering.tensors[dy].to_laid_out_tensor().tensor_list for dy in self._grad_ys])
        all_grad_xs = [tf.gradients(ys=ys, xs=xs, grad_ys=grad_ys) 
                for ys, xs, grad_ys in zip(all_ys, all_xs, all_grad_ys)]
        grad_xs = mtf.transpose_list_of_lists(all_grad_xs)
        for out, grad_x in zip(self.outputs, grad_xs):
            lowering.set_tensor_lowering(out,
                    lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(grad_x))


def GenericGradFn(forward_op, grad_y):
    return GenericGradOp(forward_op, [grad_y]).outputs


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
        #out = mtf.slicewise(max_pool, [tsr], output_shape, tsr.dtype,
        #        splittable_dims)
        return out


def MaxPool(*args, **kwargs):
    return Pooling(*args, pooling_fn=tf.nn.max_pool, **kwargs)


def AvgPool(*args, **kwargs):
    return Pooling(*args, pooling_fn=tf.nn.avg_pool, **kwargs)


