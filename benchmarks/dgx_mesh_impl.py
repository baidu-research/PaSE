import functools

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf


def allconcat_dgx(xs, devices, concat_axis):
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
    if not flag: # Fallback to ring version
        return mtf.placement_mesh_impl.allconcat_ring(xs, devices, concat_axis)

    if xs0:
        fn = lambda: tf.concat(xs0, concat_axis)
        ys_0 = mtf.parallel(d_spec0, fn)
    else:
        ys_0 = []

    if xs1:
        fn = lambda: tf.concat(xs1, concat_axis)
        ys_1 = mtf.parallel(d_spec1, fn)
    else:
        ys_1 = []

    if xs0 and xs1:
        ys = [y for y in zip(ys_0*2, ys_1*2)]
        ys = mtf.parallel(devices, tf.concat, ys, [concat_axis] * len(devices))
    else:
        ys = ys_0 + ys_1
    return ys


class DGXMeshImpl(mtf.placement_mesh_impl.PlacementMeshImpl):
    def allreduce(self, x, mesh_axes, reduction_fn_string):
        return self._collective_with_groups(
                x, mesh_axes, functools.partial(
                    mtf.placement_mesh_impl.allreduce_ring_single_shard,
                    reduction_fn_string=reduction_fn_string))

    def alltoall(self, x, mesh_axis, split_axis, concat_axis):
        return self._collective_with_groups(
                x, [mesh_axis],
                functools.partial(
                    alltoall_pointtwise, split_axis=split_axis,
                    concat_axis=concat_axis))

    def allconcat(self, x, mesh_axis, concat_axis):
        return self._collective_with_groups(
                x, [mesh_axis],
                functools.partial(allconcat_dgx, concat_axis=concat_axis))


