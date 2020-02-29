import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import mesh_tensorflow as mtf
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

    def GetMeshImpl(dev_cnts, devices=None, node_cnt=num_nodes):
        assert ((utils.RoundUp(utils.Prod(dev_cnts), gpus_per_node)) ==
                (gpus_per_node * num_nodes))
        return utils.GetMeshImpl(dev_cnts, devices=devices, num_nodes=node_cnt)

    mesh = mtf.Mesh(graph, 'mesh0')
    meshes.append(mesh)
    mesh_to_impl[mesh] = GetMeshImpl([num_gpus])

    assert len(inputs.shape) == 2
    assert inputs.shape == labels.shape

    shape = GetShape([('axis0', batch_size), inputs.get_shape().as_list()[1]])
    mtf_inputs = mtf.import_tf_tensor(mesh, inputs, shape)
    mtf_labels = mtf.import_tf_tensor(mesh, labels, shape)

    return graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels

class LSTMCell(keras.layers.Layer):
    def __init__(self, batch_size, num_units, layer, **kwargs):
        self.batch_size = batch_size
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

def model(params, inputs, labels):
    devices = params.devices
    num_gpus = len(devices)
    learning_rate = 0.01

    # RNN cells
    cells = [LSTMCell(params.batch_size, params.num_units, layer=0),
            LSTMCell(params.batch_size, params.num_units, layer=1)]
    rnn = keras.layers.RNN(cells, return_sequences=True, return_state=False)

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
    y = mtf.slicewise(rnn, [embedding], output_shape=embedding.shape,
            output_dtype=embedding.dtype, splittable_dims=embedding.shape[:1])
    y = mtf.layers.dense(y, vocab_dim, reduced_dims=y.shape[-1:],
            use_bias=False)
    mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(y, mtf_labels,
            vocab_dim)
    mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    # Optimize
    with tf.variable_scope('optimize'):
        grads = mtf.gradients([mtf_loss], [v.outputs[0] for v in
            graph.trainable_variables])
        opt = mtf.optimize.SgdOptimizer(learning_rate)
        grad_updates = opt.apply_grads(grads, graph.trainable_variables)


    print('Beginning to lower mtf graph...', flush=True)
    lowering = mtf.Lowering(graph, mesh_to_impl)
    print('Finished lowering.', flush=True)
    tf_loss = lowering.export_to_tf_tensor(mtf_loss)
    tf_grad_updates = [lowering.lowered_operation(op) for op in grad_updates]

    # Initializer
    tf_init_vars = utils.FlattenList([
                lowering.variables[var].laid_out_tensor.all_slices for var in
                graph.trainable_variables])
    init_op = tf.trainable_variables()
    for v in tf_init_vars:
        with tf.device(v.device):
            init_op.append(v.initializer)
    print(init_op)

    return init_op, tf_loss, tf_grad_updates

