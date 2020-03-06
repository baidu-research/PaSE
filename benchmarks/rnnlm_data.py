import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import mesh_tensorflow as mtf
import utils

def CreateMeshes(inputs, labels, num_nodes, num_gpus, batch_size):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}

    mesh = mtf.Mesh(graph, 'mesh0')
    meshes.append(mesh)
    mesh_to_impl[mesh] = utils.GetMeshImpl([num_gpus], num_nodes=num_nodes)

    assert len(inputs.shape) == 2
    assert inputs.shape == labels.shape

    shape = utils.ConvertToShape([('axis0', batch_size),
        inputs.shape.as_list()[1]])
    mtf_inputs = mtf.import_tf_tensor(mesh, inputs, shape)
    mtf_labels = mtf.import_tf_tensor(mesh, labels, shape)

    return graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels

class LSTMCell(keras.layers.Layer):
    def __init__(self, num_units, layer, **kwargs):
        self.num_units = num_units
        self.state_size = [num_units, num_units]
        self.layer = layer
        super().__init__(**kwargs)

    def build(self, input_state):
        w_shape = [2 * self.num_units, 4 * self.num_units]
        self.w = self.add_weight(shape=w_shape, initializer='uniform',
                name=f'w_l{self.layer}', dtype=tf.float32)
        super().build(input_state)

    def call(self, x, states):
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

class RNNGradOperation(mtf.GenericGradOperation):
    def lower(self, lowering):
        rnn_op = self._forward_op
        tf_rnn_op = rnn_op._tf_fn
        assert (len(rnn_op.inputs) == len(rnn_op.outputs) 
                == len(self._grad_ys) == 1)

        get_tensor_list = lambda x: lowering.tensors[
                x].to_laid_out_tensor().tensor_list

        ys = get_tensor_list(rnn_op.outputs[0])
        xs = get_tensor_list(rnn_op.inputs[0])
        ws = tf_rnn_op.weights
        grad_ys = get_tensor_list(self._grad_ys[0])

        # Since we perform RNN as slicewise operation, dy_i/dx_j for i!=j is zero.
        # So we only compute dy_i/dx_i for various slices
        assert len(ys) == len(xs) == len(grad_ys)
        assert len(ws) == 2
        grad_xs_ws = [tf.gradients(y, [x] + ws, grad_ys=grad_y) for y, x, grad_y in
                zip(ys, xs, grad_ys)]
        assert all(len(g) == 3 for g in grad_xs_ws)
        grad_xs, grad_w0s, grad_w1s = utils.TransposeLists(grad_xs_ws)

        # Accumulate dy_i/dw_j for different w_j
        tf_rnn_op.grad_ws = [tf.add_n(grad_w0s), tf.add_n(grad_w1s)]
        lowering.set_tensor_lowering(self.outputs[0],
                lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(
                    grad_xs))

def RNNGradFn(forward_op, grad_y):
    return RNNGradOperation(forward_op, [grad_y]).outputs

def model(params, inputs, labels):
    devices = params.devices
    num_gpus = len(devices)

    # RNN cells
    cells = [LSTMCell(params.num_units, layer=0),
             LSTMCell(params.num_units, layer=1)]
    tf_rnn_op = keras.layers.RNN(cells, return_sequences=True, return_state=False)

    # Mtf mesh
    assert len(inputs.shape) == 2
    graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels = CreateMeshes(
            inputs, labels, params.num_nodes, num_gpus, params.batch_size)

    # Embedding dimensions
    vocab_dim = mtf.Dimension(utils.RandName(), params.vocab_size)
    embed_dim = mtf.Dimension(utils.RandName(), params.num_units)

    # Model
    embedding = mtf.layers.embedding(mtf_inputs, vocab_dim, embed_dim,
            tf.float32)
    mtf_rnn = mtf.slicewise(tf_rnn_op, [embedding], output_shape=embedding.shape,
            grad_function=RNNGradFn, output_dtype=embedding.dtype,
            splittable_dims=embedding.shape[:1])
    y = mtf.layers.dense(mtf_rnn, vocab_dim, reduced_dims=mtf_rnn.shape[-1:],
            use_bias=False)
    mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(y, mtf_labels,
            vocab_dim)
    mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    return graph, mesh_to_impl, mtf_loss, tf_rnn_op
