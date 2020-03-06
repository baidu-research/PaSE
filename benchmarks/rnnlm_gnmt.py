import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import mesh_tensorflow as mtf
import utils
from rnnlm_data import RNNGradOperation

def CreateMeshes(inputs, labels, num_nodes, num_gpus, batch_size):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}

    assert num_gpus % num_nodes == 0
    gpus_per_node = num_gpus // num_nodes
    assert (num_gpus//2) % gpus_per_node == 0 # TODO: check
    num_nodes_per_mesh = (num_gpus//2) // gpus_per_node

    mesh = mtf.Mesh(graph, f'mesh0')
    meshes.append(mesh)
    mesh_to_impl[mesh] = utils.GetMeshImpl([num_gpus//2],
            num_nodes=num_nodes_per_mesh)

    mesh = mtf.Mesh(graph, f'mesh1')
    meshes.append(mesh)
    mesh_to_impl[mesh] = utils.GetMeshImpl([num_gpus//2],
            num_nodes=num_nodes_per_mesh)

    assert len(inputs.shape) == 2
    assert inputs.shape == labels.shape

    shape = utils.ConvertToShape([('axis0', batch_size),
        inputs.shape.as_list()[1]])
    mtf_inputs = mtf.import_tf_tensor(meshes[0], inputs, shape)
    mtf_labels = mtf.import_tf_tensor(meshes[-1], labels, shape)

    return graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels

class LSTMCell(keras.layers.Layer):
    def __init__(self, num_units, layer, device, **kwargs):
        self.num_units = num_units
        self.state_size = [num_units, num_units]
        self.layer = layer
        self.device = device
        super().__init__(**kwargs)

    def build(self, input_state):
        w_shape = [2 * self.num_units, 4 * self.num_units]
        with tf.device(self.device):
            self.w = self.add_weight(shape=w_shape, initializer='uniform',
                    name=f'w_l{self.layer}', dtype=tf.float32)
        super().build(input_state)

    def get_device(self, curr_device):
        base_device_id = int(self.device.split(':')[-1])
        assert base_device_id >= 0

        if base_device_id > 0:
            sp = curr_device.split(':')
            curr_device_id = int(sp[-1])
            new_device_id = base_device_id + curr_device_id
            sp[-1] = str(new_device_id)
            new_device = ':'.join(sp)
        else:
            new_device = curr_device

        return new_device

    def call(self, x, states):
        assert x.device
        device = self.get_device(x.device)

        with tf.device(device):
            h, c = states
            xh = tf.concat([x, h], axis=1)

            # GEMM
            ifgo = tf.matmul(xh, self.w)

            # Apply activations
            i, f, g, o = tf.split(ifgo, 4, axis=1)
            i, f, o = [tf.sigmoid(t) for t in (i, f, o)]
            g = tf.tanh(g)

            # Elementwise ops
            c = (f * c) + (i * g)
            h = o * tf.tanh(c)
            return h, [h, c]

class RNNOperation(mtf.Operation):
    def __init__(self, x, mesh, tf_rnn_op, name=None):
        super().__init__([x], mesh=mesh, name=name or 'rnn')
        self.outputs = [mtf.Tensor(self, x.shape, x.dtype)]
        self._tf_rnn_op = tf_rnn_op

    def gradient(self, grad_ys):
        [grad_x] = RNNGradOperation(self, grad_ys).outputs
        assert grad_x.mesh == self.inputs[0].mesh
        return [grad_x]

    def lower(self, lowering):
        x = self.inputs[0]
        input_slices = lowering.tensors[x].to_laid_out_tensor().tensor_list
        old_mesh_impl = lowering.mesh_impl(x.mesh)

        laid_out_y = old_mesh_impl.slicewise(self._tf_rnn_op, input_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_y)

def model(params, inputs, labels):
    devices = params.devices
    num_gpus = len(devices)
    num_nodes = params.num_nodes

    # RNN cells
    cells = [LSTMCell(params.num_units, 0, devices[0]),
             LSTMCell(params.num_units, 1, devices[num_gpus//2])]
    tf_rnn_op = keras.layers.RNN(cells, return_sequences=True, return_state=False)

    # Mtf mesh
    assert len(inputs.shape) == 2
    graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels = CreateMeshes(
            inputs, labels, num_nodes, num_gpus, params.batch_size)

    # Embedding dimensions
    vocab_dim = mtf.Dimension(utils.RandName(), params.vocab_size)
    embed_dim = mtf.Dimension(utils.RandName(), params.num_units)

    # Model
    embedding = mtf.layers.embedding(mtf_inputs, vocab_dim, embed_dim,
            tf.float32)
    mtf_rnn = RNNOperation(embedding, meshes[1], tf_rnn_op).outputs[0]
    y = mtf.layers.dense(mtf_rnn, vocab_dim, reduced_dims=mtf_rnn.shape[-1:],
            use_bias=False)
    assert y.mesh == mtf_labels.mesh
    mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(y, mtf_labels,
            vocab_dim)
    mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    return graph, mesh_to_impl, mtf_loss, tf_rnn_op
