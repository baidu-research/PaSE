import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import string
import utils

def GetShape(dims):
    sh = []
    for d in dims:
        try:
            name, size = d
        except (TypeError, ValueError):
            name, size = utils.RandName(), d
        sh.append(mtf.Dimension(name, size))

    sh = mtf.Shape(sh)
    return sh

def CreateMeshes(inputs, labels, num_nodes, num_gpus, batch_size):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}

    assert num_gpus % num_nodes == 0
    gpus_per_node = num_gpus // num_nodes

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

    def GetMeshImpl(dev_cnts, devices=None, node_cnt=num_nodes):
        assert ((utils.RoundUp(utils.Prod(dev_cnts), gpus_per_node)) ==
                (gpus_per_node * node_cnt))
        return utils.GetMeshImpl(dev_cnts, devices=devices, num_nodes=node_cnt)

    mesh = mtf.Mesh(graph, 'mesh0')
    meshes.append(mesh)
    mesh_to_impl[mesh] = GetMeshImpl([n, k])

    mesh = mtf.Mesh(graph, 'mesh1')
    meshes.append(mesh)
    mesh_to_impl[mesh] = GetMeshImpl([num_gpus])

    assert len(inputs.shape) == 2
    assert inputs.shape == labels.shape

    mtf_shape = GetShape(inputs.get_shape().as_list())
    mtf_inputs = mtf.import_tf_tensor(meshes[0], inputs, mtf_shape)
    mtf_labels = mtf.import_tf_tensor(meshes[1], labels, mtf_shape)

    return graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels

class LSTMCell(keras.layers.Layer):
    def __init__(self, num_units, layer, mesh_impl, **kwargs):
        self.num_units = num_units
        self.layer = layer
        self.mesh_impl = mesh_impl

        assert len(mesh_impl.shape) == 2
        self.axis_n, self.axis_k = 0, 1
        part_n = mesh_impl.shape.to_integer_list[self.axis_n]
        part_k = mesh_impl.shape.to_integer_list[self.axis_k]
        self.num_gpus = mesh_impl.size

        assert num_units % part_k == 0
        assert num_units % part_n == 0
        self.part_k_size = num_units // part_k
        self.part_n_size = num_units // part_n

        h_state_sizes = [self.part_k_size] * self.num_gpus
        c_state_sizes = [self.part_n_size] * self.num_gpus
        self.state_size = h_state_sizes + c_state_sizes
        self.output_size = h_state_sizes
        super().__init__(**kwargs)

    def build(self, input_shapes):
        w_shape = [2 * self.part_k_size, 4 * self.part_n_size]
        ws = []
        for i, dev in enumerate(self.mesh_impl.devices):
            with tf.device(dev):
                ws.append(self.add_weight(
                    shape=w_shape,
                    initializer='uniform',
                    name=f'w_l{self.layer}_part{i}',
                    dtype=tf.float32))
        self.laid_out_w = self.mesh_impl.LaidOutTensor(ws)
        super().build(input_shapes)

    def call(self, xs, states):
        mesh_impl = self.mesh_impl
        assert len(xs) == self.num_gpus
        assert len(states) == 2 * self.num_gpus
        #assert [x.device for x in xs] == mesh_impl.devices, (
        #        [x.device for x in xs], mesh_impl.devices)

        # State tensors
        hs, cs = states[:self.num_gpus], states[self.num_gpus:]
        #assert [h.device for h in hs] == mesh_impl.devices
        #assert [c.device for c in cs] == mesh_impl.devices

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
        laid_out_h, laid_out_c = mesh_impl.slicewise(act_fn, laid_out_y,
                laid_out_c)
        hs = laid_out_h.tensor_list
        cs = laid_out_c.tensor_list

        return hs, hs + cs

class RNNOperation(mtf.Operation):
    def __init__(self, x, rnn_op, name=None):
        super().__init__([x], name=name or 'rnn')
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]
        self.rnn_op = rnn_op

    def gradient(self, grad_ys):
        return mtf.GenericGradOperation(self, grad_ys).outputs

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        input_slices = lowering.tensors[
                self.inputs[0]].to_laid_out_tensor().tensor_list

        y = self.rnn_op(tuple(input_slices))
        lowering.set_tensor_lowering(self.outputs[0], y)

def rnn(x, rnn_op):
    return RNNOperation(x, rnn_op).outputs[0]

def model(params, inputs, labels):
    devices = params.devices
    num_gpus = len(devices)
    lr = 0.01

    # MTF mesh
    assert len(inputs.shape) == 2
    graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels = CreateMeshes(
            inputs, labels, params.num_nodes, num_gpus, params.batch_size)

    # Embedding dimensions
    vocab_dim = mtf.Dimension('axis0', params.vocab_size)
    embed_dim = mtf.Dimension('axis1', params.num_units)

    # RNN
    mesh_impl = mesh_to_impl[meshes[0]]
    cells = [LSTMCell(params.num_units, 0, mesh_impl), 
             LSTMCell(params.num_units, 1, mesh_impl)]
    rnn_op = keras.layers.RNN(cells, return_sequences=True, return_state=False)

    # Model
    embedding = mtf.layers.embedding(mtf_inputs, vocab_dim, embed_dim,
            tf.float32)
    assert embedding.shape[-1].name == 'axis1'

    mtf_rnn = rnn(embedding, rnn_op)
    y = mtf.layers.dense(mtf_rnn, vocab_dim, reduced_dims=mtf_rnn.shape[-1:],
            use_bias=False)

    # TODO: change mesh
    mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(y, mtf_labels,
            vocab_dim)
    mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    # Optimize
    mtf_trainable_vars = [v.outputs[0] for v in graph.trainable_variables]
    *grads, rnn_grad = mtf.gradients([mtf_loss], mtf_trainable_vars + [mtf_rnn])
    opt = mtf.optimize.SgdOptimizer(lr)
    grad_updates = opt.apply_grads(grads, graph.trainable_variables)

    # Lower
    print('Beginning to lower mtf graph...', flush=True)
    lowering = mtf.Lowering(graph, mesh_to_impl)
    print('Finished lowering.', flush=True)

    # Loss and gradients
    tf_loss = lowering.export_to_tf_tensor(mtf_loss)
    tf_grad_updates = [lowering.lowered_operation(op) for op in grad_updates]

    # RNN weight update
    tf_rnn = lowering.tensors[mtf_rnn].tensor_list
    tf_rnn_vars = rnn_op.weights
    tf_rnn_output_grads = lowering.tensors[rnn_grad].tensor_list
    assert len(tf_rnn) == len(tf_rnn_output_grads)
    tf_rnn_grads = tf.gradients(tf_rnn, tf_rnn_vars, grad_ys=tf_rnn_output_grads)
    assert len(tf_rnn_grads) == len(tf_rnn_vars)
    tf_grad_updates += [tf.assign_sub(v, lr * g) for v, g in zip(tf_rnn_vars,
        tf_rnn_grads)]

    init_op = lowering.copy_masters_to_slices()
    return init_op, tf_loss, tf_grad_updates

