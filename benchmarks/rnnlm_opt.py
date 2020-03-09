import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import string
import utils
from mesh_transformations import ReplaceMesh
import mtf_operations as mt

def CreateMeshes(inputs, labels, num_nodes, num_gpus, batch_size):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}

    if num_gpus == 4:
        n, k = 2, 2
    elif num_gpus == 8:
        n, k = 4, 2
    elif num_gpus == 16:
        n, k = 4, 4
    elif num_gpus == 32:
        n, k = 8, 4
    elif num_gpus == 64:
        n, k = 8, 8
    else:
        assert False
    assert (n*k) == num_gpus

    mesh = mtf.Mesh(graph, 'mesh0')
    meshes.append(mesh)
    mesh_to_impl[mesh] = utils.GetMeshImpl([n, k], num_nodes=num_nodes)

    mesh = mtf.Mesh(graph, 'mesh1')
    meshes.append(mesh)
    mesh_to_impl[mesh] = utils.GetMeshImpl([num_gpus], num_nodes=num_nodes)

    assert len(inputs.shape) == 2
    assert inputs.shape == labels.shape

    mtf_shape = utils.ConvertToShape(inputs.shape.as_list())
    mtf_inputs = mtf.import_tf_tensor(meshes[0], inputs, mtf_shape)
    mtf_labels = mtf.import_tf_tensor(meshes[1], labels, mtf_shape)

    return graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels

class LSTMCell(keras.layers.Layer):
    def __init__(self, num_units, ws, mesh_impl, **kwargs):
        self.num_units = num_units
        self.laid_out_w = ws
        self.mesh_impl = mesh_impl

        assert len(mesh_impl.shape) == 2
        self.axis_n, self.axis_k = 0, 1
        part_n = mesh_impl.shape.to_integer_list[self.axis_n]
        part_k = mesh_impl.shape.to_integer_list[self.axis_k]
        self.num_gpus = mesh_impl.size

        assert num_units % part_k == 0
        assert num_units % part_n == 0
        part_k_size = num_units // part_k
        part_n_size = num_units // part_n

        h_state_sizes = [part_k_size] * self.num_gpus
        c_state_sizes = [part_n_size] * self.num_gpus
        self.state_size = h_state_sizes + c_state_sizes
        self.output_size = h_state_sizes
        super().__init__(**kwargs)

    def call(self, xs, states):
        mesh_impl = self.mesh_impl
        assert len(xs) == self.num_gpus
        assert len(states) == 2 * self.num_gpus

        # State tensors
        hs, cs = states[:self.num_gpus], states[self.num_gpus:]

        # Laid out tensors
        laid_out_x = mesh_impl.LaidOutTensor(list(xs))
        laid_out_h = mesh_impl.LaidOutTensor(list(hs))
        laid_out_c = mesh_impl.LaidOutTensor(list(cs))

        # Concat x and h
        concat_fn = lambda x, y: tf.concat([x, y], axis=1)
        laid_out_xh = mesh_impl.slicewise(concat_fn, laid_out_x, laid_out_h)

        # GEMM: y = xh * w
        partial_y = mesh_impl.slicewise(tf.matmul, laid_out_xh, self.laid_out_w)
        laid_out_y = mesh_impl.allreduce(partial_y, [self.axis_k], "SUM")

        def act_fn(x, c):
            # Activations
            i, f, g, o = tf.split(x, 4, axis=1)
            i, f, o = [tf.sigmoid(t) for t in (i, f, o)]
            g = tf.tanh(g)

            # Elementwise ops
            c = (f * c) + (i * g)
            h = o * tf.tanh(c)

            return h, c

        # Apply activation and elementwise ops
        # We use the trick of (implicitly) shuffling columns of weight matrix,
        # so that the correct corresponding slices of 'i,f,g,o' end up on same
        # devices, so that no communication is necessary for elementwise
        # operations
        laid_out_h, laid_out_c = mesh_impl.slicewise(act_fn, laid_out_y,
                laid_out_c)

        # Map last dim of 'hs' from 'axis0' to 'axis1'
        laid_out_h = mesh_impl.allconcat(laid_out_h, self.axis_n, 1)
        laid_out_h = mesh_impl.allsplit(laid_out_h, self.axis_k, 1)

        hs = laid_out_h.tensor_list
        cs = laid_out_c.tensor_list
        return hs, hs + cs

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

        grad_xs_ws = tf.gradients(ys, xs + ws_l0 + ws_l1, grad_ys=grad_ys,
                colocate_gradients_with_ops=True)
        assert len(grad_xs_ws) == (len(xs) + len(ws_l0) + len(ws_l1))

        grad_xs = grad_xs_ws[:len(xs)]
        grad_ws_l0 = grad_xs_ws[len(xs):len(xs)+len(ws_l0)]
        grad_ws_l1 = grad_xs_ws[len(xs)+len(ws_l0):]

        # Laid out tensors
        grad_xs_lo = mesh_impl.LaidOutTensor.from_tensor_list(grad_xs)
        grad_ws_l0_lo = mesh_impl.LaidOutTensor.from_tensor_list(grad_ws_l0)
        grad_ws_l1_lo = mesh_impl.LaidOutTensor.from_tensor_list(grad_ws_l1)

        assert len(self.outputs) == 3
        lowering.set_tensor_lowering(self.outputs[0], grad_xs_lo)
        lowering.set_tensor_lowering(self.outputs[1], grad_ws_l0_lo)
        lowering.set_tensor_lowering(self.outputs[2], grad_ws_l1_lo)

class RNNOperation(mtf.Operation):
    def __init__(self, x, w0, w1, num_units, name=None):
        assert x.shape[-1].name == w0.shape[0].name == w1.shape[0].name
        super().__init__([x, w0, w1], name=name or 'rnn')
        self.num_units = num_units
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def gradient(self, grad_ys):
        return RNNGradOperation(self, grad_ys, name='rnn_grad').outputs

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        x, w0, w1 = [lowering.tensors[x] for x in self.inputs]
        x, w0, w1 = mtf.convert_args_to_laid_out_tensors([x, w0, w1])

        cells = [LSTMCell(self.num_units, w0, mesh_impl),
                LSTMCell(self.num_units, w1, mesh_impl)]
        tf_rnn_op = keras.layers.RNN(cells, return_sequences=True,
                return_state=False)

        ys = tf_rnn_op(tuple(x.tensor_list))
        assert len(ys) == len(mesh_impl.devices)

        laid_out_y = mesh_impl.LaidOutTensor.from_tensor_list(list(ys))
        lowering.set_tensor_lowering(self.outputs[0], laid_out_y)

def model(params, inputs, labels):
    devices = params.devices
    num_gpus = len(devices)

    # MTF mesh
    assert len(inputs.shape) == 2
    graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels = CreateMeshes(
            inputs, labels, params.num_nodes, num_gpus, params.batch_size)

    # RNN weights
    num_units = params.num_units
    w_shape = utils.ConvertToShape([('axis1', 2*num_units),
        ('axis0', 4*num_units)])
    rnn_w0 = mtf.get_variable(meshes[0], 'rnn_w0', w_shape)
    rnn_w1 = mtf.get_variable(meshes[0], 'rnn_w1', w_shape)

    # Embedding dimensions
    vocab_dim = mtf.Dimension('axis0', params.vocab_size)
    embed_dim = mtf.Dimension('axis1', params.num_units)

    # Model
    embedding = mtf.layers.embedding(mtf_inputs, vocab_dim, embed_dim,
            tf.float32)
    assert embedding.shape[-1].name == 'axis1'
    [y] = RNNOperation(embedding, rnn_w0, rnn_w1, num_units).outputs

    assert y.shape[-1].name == 'axis1'
    y = mt.RenameDimension(y, 'axis1', utils.RandName())
    y = ReplaceMesh(y, meshes[1])

    y = mtf.layers.dense(y, vocab_dim, reduced_dims=y.shape[-1:],
            use_bias=False)
    mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(y, mtf_labels,
            vocab_dim)
    mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    return graph, mesh_to_impl, mtf_loss
