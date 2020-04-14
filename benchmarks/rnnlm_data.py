import numpy as np
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
import utils
from rnnlm import RNNOperation

def CreateMeshes(inputs, labels, num_nodes, num_gpus, batch_size):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}

    mesh = mtf.Mesh(graph, 'mesh0')
    meshes.append(mesh)
    mesh_to_impl[mesh] = utils.GetMeshImpl([num_gpus], 
            gpus_per_node=num_gpus // num_nodes)

    assert len(inputs.shape) == 2
    assert inputs.shape == labels.shape

    shape = utils.ConvertToShape([('axis0', batch_size),
        inputs.shape.as_list()[1]])
    mtf_inputs = mtf.import_tf_tensor(mesh, inputs, shape)
    mtf_labels = mtf.import_tf_tensor(mesh, labels, shape)

    return graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels

'''
class LSTMCell(keras.layers.Layer):
    def __init__(self, num_units, ws, **kwargs):
        self.num_units = num_units
        self.ws = {w.device:w for w in ws}
        self.state_size = [num_units, num_units]
        super().__init__(**kwargs)

    def call(self, x, states):
        assert x.device
        h, c = states
        xh = tf.concat([x, h], axis=1)

        # GEMM
        ifgo = tf.matmul(xh, self.ws[x.device])

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
        mesh_impl = lowering.mesh_impl(self)
        rnn_op = self._forward_op

        get_tensor_list = lambda x: lowering.tensors[
                x].to_laid_out_tensor().tensor_list

        assert len(rnn_op.inputs) == 3
        assert len(rnn_op.outputs) == 1
        assert len(self._grad_ys) == 1
        xs = get_tensor_list(rnn_op.inputs[0])
        ws_l0 = get_tensor_list(rnn_op.inputs[1])
        ws_l1 = get_tensor_list(rnn_op.inputs[2])
        ys = get_tensor_list(rnn_op.outputs[0])
        grad_ys = get_tensor_list(self._grad_ys[0])

        # Since we perform RNN as slicewise operation, dy_i/dx_j for i!=j is zero.
        # So we only compute dy_i/dx_i for various slices. (Replicated) weights
        # are all-reduced separately below.
        assert (len(ys) == len(xs) == len(grad_ys) == len(ws_l0) == len(ws_l1))
        grad_xs_ws = [tf.gradients(y, [x, w0, w1], grad_ys=grad_y,
            colocate_gradients_with_ops=True) for y, x, w0, w1, grad_y in
            zip(ys, xs, ws_l0, ws_l1, grad_ys)]
        assert all(len(g) == 3 for g in grad_xs_ws)
        grad_xs, grad_ws_l0, grad_ws_l1 = utils.TransposeLists(grad_xs_ws)

        # Laid out tensors
        grad_xs_lo = mesh_impl.LaidOutTensor.from_tensor_list(grad_xs)
        grad_ws_l0_lo = mesh_impl.LaidOutTensor.from_tensor_list(grad_ws_l0)
        grad_ws_l1_lo = mesh_impl.LaidOutTensor.from_tensor_list(grad_ws_l1)

        # Accumulate dy_i/dw_j for replicated w_j's
        grad_ws_l0_lo = mesh_impl.allreduce(grad_ws_l0_lo, [0], 'SUM')
        grad_ws_l1_lo = mesh_impl.allreduce(grad_ws_l1_lo, [0], 'SUM')

        lowering.set_tensor_lowering(self.outputs[0], grad_xs_lo)
        lowering.set_tensor_lowering(self.outputs[1], grad_ws_l0_lo)
        lowering.set_tensor_lowering(self.outputs[2], grad_ws_l1_lo)

class RNNOperation(mtf.Operation):
    def __init__(self, x, w0, w1, num_units, name=None):
        super().__init__([x, w0, w1], name=name or 'rnn')
        self.num_units = num_units
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def gradient(self, grad_ys):
        return RNNGradOperation(self, grad_ys, name='rnn_grad').outputs

    def lower(self, lowering):
        x, w0, w1 = [lowering.tensors[x] for x in self.inputs]
        w0_lo, w1_lo = [w.tensor_list for w in
                mtf.convert_args_to_laid_out_tensors([w0, w1])]

        cells = [LSTMCell(self.num_units, w0_lo),
                LSTMCell(self.num_units, w1_lo)]
        tf_rnn_op = keras.layers.RNN(cells, return_sequences=True,
                return_state=False)

        y = lowering.mesh_impl(self).slicewise(tf_rnn_op, x)
        lowering.set_tensor_lowering(self.outputs[0], y)
'''

def model(params, inputs, labels):
    # Mtf mesh
    assert len(inputs.shape) == 2
    graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels = CreateMeshes(
            inputs, labels, params.num_nodes, params.num_gpus,
            params.batch_size)

    # Embedding dimensions
    vocab_dim = mtf.Dimension(utils.RandName(), params.vocab_size)
    embed_dim = mtf.Dimension(utils.RandName(), params.num_units)

    batch_dim_name = mtf_inputs.shape[0].name
    k_dim_name = embed_dim.name
    n_dim_name = utils.RandName()

    # RNN weights
    num_units = params.num_units
    w_shape = utils.ConvertToShape(
            [(k_dim_name, 2*num_units), (n_dim_name, 4*num_units)])
    rnn_w0 = mtf.get_variable(meshes[0], 'rnn_w0', w_shape)
    rnn_w1 = mtf.get_variable(meshes[0], 'rnn_w1', w_shape)

    # RNN initial states
    h_shape = mtf.Shape([mtf.Dimension(batch_dim_name, params.batch_size),
        mtf.Dimension(k_dim_name, num_units)])
    c_shape = mtf.Shape([mtf.Dimension(batch_dim_name, params.batch_size),
        mtf.Dimension(n_dim_name, num_units)])
    states0 = [mtf.zeros(meshes[0], h_shape), mtf.zeros(meshes[0], c_shape)]
    states1 = [mtf.zeros(meshes[0], h_shape), mtf.zeros(meshes[0], c_shape)]

    # Model
    embedding = mtf.layers.embedding(mtf_inputs, vocab_dim, embed_dim,
            tf.float32)
    [y] = RNNOperation(embedding, rnn_w0, rnn_w1, num_units,
            states=states0+states1).outputs
    y = mtf.layers.dense(y, vocab_dim, reduced_dims=y.shape[-1:],
            use_bias=False)
    mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(y, mtf_labels,
            vocab_dim)
    mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    return graph, mesh_to_impl, mtf_loss
