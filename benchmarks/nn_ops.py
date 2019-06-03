import numpy as np
import tensorflow as tf

from functools import reduce
import operator as op


def GetGPUDevice(d):
    return tf.device(tf.DeviceSpec(device_type='GPU', device_index=d))


def TransposeLists(l):
    return [list(x) for x in zip(*l)]


def Parallelize(fn, *args, **kwargs, devices):
    n_d = len(devices)

    assert all(len(v) == n_d for v in args)
    assert all(len(v) == n_d for v in kwargs.values())

    ret = []
    for i, d in enumerate(devices):
        with GetGPUDevice(d):
            with tf.variable_scope('par_%d' % i):
                local_args = [a[i] for a in args]
                local_kwargs = {k: v[i] for k, v in kwargs.items()}
                ret.append(fn(*local_args, **local_kwargs))

    return ret


def AllReduce(tsrs, devices, reduce_fn=tf.add):
    n = len(devices)
    assert len(tsrs) == n

    if n == 1:
        return tsrs

    def AllReduceSingleShard(tsrs, devices):
        if n % 2 == 0:
            left_center = n / 2 - 1
            right_center = left_center + 1
        else:
            left_center = n // 2
            right_center = left_center

        # Reduce among the left half
        left_sum = tsrs[0]
        for i in range(1, left_center + 1):
            with GetGPUDevice(devices[i]):
                left_sum = reduce_fn(left_sum, tsrs[i])

        # Reduce among the right half
        right_sum = tsrs[n - 1]
        for i in range(n - 2, left_center, -1):
            with GetGPUDevice(devices[i]):
                right_sum = reduce_fn(tsrs[i], right_sum)

        # Reduce the two halves
        with GetGPUDevice(devices[left_center]):
            result[left_center] = reduce_fn(left_sum, right_sum)
        if n % 2 == 0:
            with GetGPUDevice(devices[right_center]):
                result[right_center] = reduce_fn(left_sum, right_sum)

        # Copy the reduced values to other devices
        for i in range(left_center - 1, -1, -1):
            with GetGPUDevice(devices[i]):
                result[i] = tf.identity(result[i + 1])
        for i in range(right_center + 1, n):
            with GetGPUDevice(devices[i]):
                result[i] = tf.identity(result[i - 1])

        return result

    shape = tsrs[0].shape.as_list()
    size = sum(shape)
    if size is None or size < 1024 or size % n != 0:
        return AllReduceSingleShard(tsrs, devices)

    def _circular_shift(l, s):
        s %= len(l)
        return l[-s:] + l[:-s]
    def _flatten_and_split(x):
        return tf.split(tf.reshape(x, [-1]), n)
    def _concat_and_reshape(l):
        return tf.reshape(tf.concat(l, 0), shape)

    # Split the tensors into multiple shards
    x_split = Parallelize(_flatten_and_split, tsrs, devices)
    x_split_t = TransposeLists(x_split)

    # Reduce different shards 
    y_split_t = []
    for shard in range(n):
      shard_xs = _circular_shift(x_split_t[shard], shard)
      shard_devices = _circular_shift(devices, shard)
      shard_ys = AllReduceSingleShard(shard_xs, shard_devices)
      y_split_t.append(_circular_shift(shard_ys, -shard))
    y_split = TransposeLists(y_split_t)

    # Concatenate the shards
    ys = Parallelize(_concat_and_reshape, y_split, devices)
    return ys


# Concatenate and replicate tensors on all devices
def AllConcat(tsrs, devices, axis):
    n = len(tsrs)
    if n == 1:
        return tsrs

    parts = [[tsrs[target] if target == source else None for source in xrange(n)]
             for target in xrange(n)]

    for distance in range(1, n / 2 + 1):
        for target in range(n):
            source = (target + distance) % n
            if parts[target][source] is None:
                with GetGPUDevice(devices[target]):
                    parts[target][source] = tf.identity(parts[(target + 1) % n][source])

            source = (target - distance) % n
            if parts[target][source] is None:
                with GetGPUDevice(devices[target]):
                    parts[target][source] = tf.identity(parts[(target - 1) % n][source])

    return Parallelize(tf.concat, parts, axis=[axis] * n)


def Conv(img, r, s, n, stride, pad, dev_b, dev_n, name):
    n_b = len(dev_b)
    n_n = len(dev_n)
    c = img.shape[-1]

    img_split = tf.split(img, n_b, axis=0)

    for d_b in dev_b:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

    for d_n in dev_n:
        weights = tf.get_variable('weights_%d' % d_n, shape = [r, s, c, n /
            len(dev_n)])

        conv = tf.nn.conv2d(img, weights, stride)

