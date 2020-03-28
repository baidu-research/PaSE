from operator import mul
from functools import reduce
import string, random

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

def is_power_of_2(v):
    return ((v == 1) or not (v & (v-1)))

def RandName(k=5):
    return ''.join(random.choices(string.ascii_letters + string.ascii_uppercase
        + string.digits, k=k))

def Prod(lst):
    return reduce(mul, lst, 1)

def RoundUp(n, m):
    assert n > 0 and m > 0
    rem = n % m
    return (n + m - rem) if rem else n

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

def ConvertToShape(dims):
    sh = []
    for d in dims:
        try:
            name, size = d
        except (TypeError, ValueError):
            name, size = RandName(), d
        sh.append(mtf.Dimension(name, size))

    sh = mtf.Shape(sh)
    return sh

def GetDeviceStr(node_id, gpu_id):
    return f'/job:localhost/replica:0/task:{node_id}/device:GPU:{gpu_id}'

def GetDeviceList(gpus, num_nodes=1, gpus_per_node=8):
    if isinstance(gpus, list):
        assert all(isinstance(g, str) for g in gpus)
        return gpus

    #num_nodes = (num_devs + gpus_per_node - 1) // gpus_per_node
    assert (((num_nodes-1)*gpus_per_node) < gpus <=
            (num_nodes*gpus_per_node)), (
                    f'Mismatch in node count. nodes: {num_nodes}; '
                    f'gpus_per_node: {gpus_per_node}; '
                    f'Total gpus: {gpus}.')

    assert gpus % num_nodes == 0
    return [GetDeviceStr(i, j) for i in range(num_nodes) for j in
            range(gpus_per_node)]

def DeviceIndex(gpu):
    assert gpu
    return gpu.device_index if isinstance(gpu,
            tf.DeviceSpec) else int(gpu.split(':')[-1])

def AssignLayout(ta_axes, mesh_axis):
    layout = []
    for a in ta_axes:
        a = a.name if isinstance(a, mtf.Dimension) else a
        layout.append((a, mesh_axis))
    return layout

def RenameDim(shape, axis, name):
    assert isinstance(shape, mtf.Shape)
    return shape.rename_dimension(shape[axis].name, name)

def RenameDims(shape, axes, names):
    assert len(axes) == len(names)
    for axis, name in zip(axes, names):
        shape = RenameDim(shape, axis, name)
    return shape

def GetMeshImpl(dev_cnts, devices=None, axes=None, mesh_impl=None, num_nodes=1,
        gpus_per_node=8):
    num_devs = Prod(dev_cnts)
    assert num_devs % num_nodes == 0, (
            f'Device count is not a multiple of node count. '
            f'Device count: {num_devs}; node count: {num_nodes}.')

    mesh_impl = mesh_impl or mtf.placement_mesh_impl.PlacementMeshImpl
    axes = axes or ['axis%d' % i for i in range(len(dev_cnts))]
    assert len(dev_cnts) == len(axes)

    mesh_shape = []
    layout_rules = []
    for d, axis in zip(dev_cnts, axes):
        mesh_shape.append((axis, d))
        layout_rules.append((axis, axis))

    devices = GetDeviceList(devices or num_devs, num_nodes, gpus_per_node)
    return mesh_impl(mesh_shape, layout_rules, devices)

def Optimize(graph, loss, lr=0.01):
    grads = mtf.gradients([loss], [v.outputs[0] for v in
        graph.trainable_variables])
    assert all(g for g in grads)
    opt = mtf.optimize.SgdOptimizer(lr)
    return opt.apply_grads(grads, graph.trainable_variables)

def join_tasks(task_id, hostlist, port=3452):
    if len(hostlist) <= 1:
        return

    import socket
    hostname = socket.gethostname().split('.')[0]
    assert hostname in hostlist

    def connect_and_send(hostname):
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                conn.connect((hostname, port))
                conn.close()
                break
            except (ConnectionRefusedError, FileNotFoundError) as e:
                pass

    if task_id == 0:
        for host in hostlist:
            if host != hostname:
                connect_and_send(host)
    else:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.bind(('', port))
        conn.listen(1)
        conn.accept()
        conn.close()

#def join_tasks(sess, task_id, num_tasks):
#    if num_tasks <= 1:
#        return
#
#    device = '/job:worker/replica:0/task:' + str(task_id) + '/device:CPU:0'
#    with tf.device(device):
#        if task_id == 0:
#            # Signal other workers that the task is done by enqueueing into 'q'
#            print('Task completed. Sending signal to other workers to terminate.')
#            for i in range(1, num_tasks):
#                q = tf.FIFOQueue(1, tf.int32, shared_name='worker_queue_' + str(i))
#                sess.run(q.enqueue(1))
#
#        else:
#            # Wait for the signal from task 0
#            q = tf.FIFOQueue(1, tf.int32, shared_name='worker_queue_' +
#                    str(task_id))
#            print('Worker ' + str(task_id) + ' waiting for signal.')
#            sess.run(q.dequeue())
#            print('Worker ' + str(task_id) + ' terminating.')
