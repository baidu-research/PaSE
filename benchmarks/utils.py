import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf


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


def MaxPool(tsr, fltr, stride=(1,1), padding='VALID', name=None):
    stride, padding = NormalizeStrideAndPad(stride, padding)
    with tf.variable_scope(name, default_name='pool'):
        def max_pool(x):
            return tf.nn.max_pool(x, [1, fltr[0], fltr[1], 1], [1, stride[0],
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
        out = mtf.slicewise(max_pool, [tsr], output_shape, tsr.dtype, splittable_dims)
        return out


