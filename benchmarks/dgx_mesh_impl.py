import functools

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

import utils

def split_inputs(xs, devices):
    device_ids = [d.device_index for d in devices]
    d_spec0, d_spec1, d0, d1, xs0, xs1 = [], [], [], [], [], []
    for x, spec, d in zip(xs, devices, device_ids):
        if d < 4:
            d0.append(d)
            xs0.append(x)
            d_spec0.append(spec)
        else:
            d1.append(d)
            xs1.append(x)
            d_spec1.append(spec)

    flag = (not d0) or (not d1) \
            or (len(d0) == len(d1) 
                    and all(i+4 == j and device_ids.index(i) <
                        device_ids.index(j) for i, j in zip(d0, d1)))
    return d_spec0, d_spec1, d0, d1, xs0, xs1, flag


def allconcat_dgx(xs, devices, concat_axis):
    if len(xs) == 1:
        return xs

    d_spec0, d_spec1, d0, d1, xs0, xs1, flag = split_inputs(xs, devices)
    if not flag: # Fallback to ring version
        return mtf.placement_mesh_impl.allconcat_ring(xs, devices, concat_axis)

    if xs0:
        fn = lambda: tf.concat(xs0, concat_axis)
        ys0 = mtf.parallel(d_spec0, fn)
    else:
        ys0 = []

    if xs1:
        fn = lambda: tf.concat(xs1, concat_axis)
        ys1 = mtf.parallel(d_spec1, fn)
    else:
        ys1 = []

    if xs0 and xs1:
        ys = [y for y in zip(ys0*2, ys1*2)]
        ys = mtf.parallel(devices, tf.concat, ys, [concat_axis] * len(devices))
    else:
        ys = ys0 + ys1
    return ys


def allreduce_dgx(xs, devices, reduction_fn_string="SUM"):
    if len(xs) == 1:
        return xs

    shape = xs[0].shape
    shape_list = shape.as_list()
    size = None if None in shape_list else mtf.list_product(shape_list)
    if size is None or size < 1024 or size % len(xs) != 0:
        no_split = True
    else:
        no_split = False

    d_spec0, d_spec1, d0, d1, xs0, xs1, flag = split_inputs(xs, devices)
    if not flag: # Fallback to ring version
        return mtf.placement_mesh_impl.allreduce_ring(xs, devices,
                reduction_fn_string)

    binary_reduction = mtf.binary_reduction_fn(reduction_fn_string)

    def reduce(xs, d_spec):
        if len(xs) == 1:
            return xs[0]

        def split_fn(x):
            x = tf.reshape(x, [-1])
            return tf.split(x, len(xs)) if not no_split else [x] * len(xs)
        split_xs = mtf.parallel(d_spec, split_fn, xs)
        split_xs = utils.TransposeLists(split_xs)

        def red_fn(xs):
            y = binary_reduction(xs[0], xs[1])
            for x in xs[2:]:
                y = binary_reduction(y, x)
            return y

        ys = mtf.parallel(d_spec, red_fn, split_xs)
        ys = allconcat_dgx(ys, d_spec, 0) if not no_split else ys
        return ys

    ys0 = reduce(xs0, d_spec0) if xs0 else []
    ys1 = reduce(xs1, d_spec1) if xs1 else []

    if xs0 and xs1:
        ys = []
        for x0, x1, s0, s1 in zip(ys0, ys1, d_spec0, d_spec1):
            ys += reduce([x0, x1], [s0, s1])
    else:
        ys = ys0 + ys1

    fn = lambda x: tf.reshape(x, shape)
    ys = mtf.parallel(devices, fn, ys)
    return ys


class DGXMeshImpl(mtf.placement_mesh_impl.PlacementMeshImpl):
    def allreduce(self, x, mesh_axes, reduction_fn_string):
        return self._collective_with_groups(
                x, mesh_axes, functools.partial(
                    allreduce_dgx,
                    reduction_fn_string=reduction_fn_string))

    #def alltoall(self, x, mesh_axis, split_axis, concat_axis):
    #    return self._collective_with_groups(
    #            x, [mesh_axis],
    #            functools.partial(
    #                alltoall_pointtwise, split_axis=split_axis,
    #                concat_axis=concat_axis))

    def allconcat(self, x, mesh_axis, concat_axis):
        return self._collective_with_groups(
                x, [mesh_axis],
                functools.partial(allconcat_dgx, concat_axis=concat_axis))


