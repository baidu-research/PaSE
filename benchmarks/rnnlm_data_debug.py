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
    def __init__(self, num_units, ws, layer, **kwargs):
        self.num_units = num_units
        self.ws = {w.device:w for w in ws}
        self.state_size = [num_units, num_units]
        self.layer = layer
        super().__init__(**kwargs)

    def build(self, input_shape):
        w_shape = [2 * self.num_units, 4 * self.num_units]
        self.w = self.add_weight(shape=w_shape, initializer='uniform',
                name=f'w_l{self.layer}', dtype=tf.float32)
        super().build(input_shape)

    def call(self, x, states):
        assert x.device
        h, c = states
        xh = tf.concat([x, h], axis=1)

        # GEMM
        with tf.control_dependencies([tf.assert_near(self.ws[x.device],
            self.w)]):
            ifgo1 = tf.matmul(xh, self.ws[x.device])
            ifgo2 = tf.matmul(xh, self.w)

        ifgo1 = tf.split(ifgo1, 4, axis=1)
        ifgo2 = tf.split(ifgo2, 4, axis=1)
        i, g, f, o = [(x+y)/2 for x, y in zip(ifgo1, ifgo2)]

        # Apply activations
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
        grad_ws_l0_lo = mesh_impl.LaidOutTensor.from_tensor_list(grad_ws_l0)
        grad_ws_l1_lo = mesh_impl.LaidOutTensor.from_tensor_list(grad_ws_l1)

        # Accumulate dy_i/dw_j for replicated w_j's
        grad_ws_l0_lo = mesh_impl.allreduce(grad_ws_l0_lo, [0], 'SUM')
        grad_ws_l1_lo = mesh_impl.allreduce(grad_ws_l1_lo, [0], 'SUM')

        grad_ws_l0 = grad_ws_l0_lo.to_laid_out_tensor().tensor_list
        grad_ws_l1 = grad_ws_l1_lo.to_laid_out_tensor().tensor_list
        tf_grads = tf.gradients(ys, rnn_op.tf_ws, grad_ys=grad_ys,
                colocate_gradients_with_ops=True)
        assert len(tf_grads) == 2
        tf_asserts = []
        tf_asserts.append(tf.assert_near(tf_grads[0], grad_ws_l0[0]))
        tf_asserts.append(tf.assert_near(tf_grads[1], grad_ws_l1[0]))
        #tf_asserts.append(tf.print(tf_grads[0]))
        #tf_asserts.append(tf.print(tf_grads[1]))
        grp = tf.group(tf_asserts)
        with tf.control_dependencies([grp]):
            xs = []
            for x in grad_xs:
                with tf.device(x.device):
                    xs.append(tf.identity(x))
            grad_xs_lo = mesh_impl.LaidOutTensor.from_tensor_list(xs)

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

        cells = [LSTMCell(self.num_units, w0_lo, 0),
                LSTMCell(self.num_units, w1_lo, 1)]
        tf_rnn_op = keras.layers.RNN(cells, return_sequences=True,
                return_state=False)
        y = lowering.mesh_impl(self).slicewise(tf_rnn_op, x)

        grp = tf.group([tf.assign(cells[0].w, w0_lo[0]),
            tf.assign(cells[1].w, w1_lo[0])])
        with tf.control_dependencies([grp]):
            ys = []
            for x in y.to_laid_out_tensor().tensor_list:
                with tf.device(x.device):
                    ys.append(tf.identity(x))
            y = lowering.mesh_impl(self).LaidOutTensor(ys)

        self.tf_ws = [cell.w for cell in cells]
        lowering.set_tensor_lowering(self.outputs[0], y)

def model(params, inputs, labels):
    num_gpus = len(params.devices)

    # Mtf mesh
    assert len(inputs.shape) == 2
    graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels = CreateMeshes(
            inputs, labels, params.num_nodes, num_gpus, params.batch_size)

    # RNN weights
    num_units = params.num_units
    w_shape = utils.ConvertToShape([2*num_units, 4*num_units])
    rnn_w0 = mtf.get_variable(meshes[0], 'rnn_w0', w_shape)
    rnn_w1 = mtf.get_variable(meshes[0], 'rnn_w1', w_shape)

    # Embedding dimensions
    vocab_dim = mtf.Dimension(utils.RandName(), params.vocab_size)
    embed_dim = mtf.Dimension(utils.RandName(), params.num_units)

    # Model
    embedding = mtf.layers.embedding(mtf_inputs, vocab_dim, embed_dim,
            tf.float32)
    [y] = RNNOperation(embedding, rnn_w0, rnn_w1, num_units).outputs
    y = mtf.layers.dense(y, vocab_dim, reduced_dims=y.shape[-1:],
            use_bias=False)
    mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(y, mtf_labels,
            vocab_dim)
    mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    return graph, mesh_to_impl, mtf_loss
