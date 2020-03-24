import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import string
import utils
import mesh_transformations as mesh_trans
import mtf_operations as mt

def assign_device(x, d):
    if x.device != d:
        with tf.device(d):
            return tf.identity(x)
    else:
        return x

def CreateMeshes(inputs, labels, num_nodes, num_gpus, batch_size):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}
    all_devices = utils.GetDeviceList(num_gpus, num_nodes)

    assert len(inputs.shape) == 2
    assert inputs.shape == labels.shape

    if num_gpus == 4:
        # Mesh_shape: batch_dim, n_dim, k_dim
        mesh_shapes = [[1, 1, 4],
                       [2, 1, 1],
                       [2, 1, 1],
                       [2, 2, 1]]

    elif num_gpus == 8:
        # Mesh_shape: batch_dim, n_dim, k_dim
        mesh_shapes = [[1, 1, 8],
                       [4, 1, 1],
                       [4, 1, 1],
                       [1, 8, 1]]

    elif num_gpus == 16:
        # Mesh_shape: batch_dim, n_dim, k_dim
        mesh_shapes = [[2, 1, 8],
                       [2, 2, 2],
                       [2, 2, 2],
                       [2, 8, 1]]

    elif num_gpus == 32:
        # Mesh_shape: batch_dim, n_dim, k_dim
        mesh_shapes = [[2, 1, 16],
                       [4, 2, 2],
                       [4, 2, 2],
                       [4, 8, 1]]

    elif num_gpus == 64:
        # Mesh_shape: batch_dim, n_dim, k_dim
        mesh_shapes = [[4, 1, 16],
                       [8, 2, 2],
                       [8, 2, 2],
                       [4, 16, 1]]

    else:
        assert False

    assert mesh_shapes[1] == mesh_shapes[2]
    assert (utils.Prod(mesh_shapes[1]) == utils.Prod(mesh_shapes[2]) ==
            num_gpus // 2)
    assert (num_nodes == 1) or (num_nodes % 2 == 0)
    half_devices0 = all_devices[:(num_gpus // 2)]
    half_devices1 = all_devices[(num_gpus // 2):]
    ## Permute the devices, so that devices are arranged in such a way that batch
    ## dim is not the last mesh dimension
    #permuted_devices = utils.FlattenList(utils.TransposeLists(
    #    [half_devices0, half_devices1]))
    #devices = [permuted_devices, half_devices0, half_devices1, permuted_devices]
    devices = [all_devices, half_devices0, half_devices1,
            half_devices1 + half_devices0]
    node_counts = [num_nodes, max(1, num_nodes // 2),
            max(1, num_nodes // 2), num_nodes]

    for i, (mesh_shape, ds, n) in enumerate(
            zip(mesh_shapes, devices, node_counts)):
        mesh = mtf.Mesh(graph, 'mesh' + str(i))
        meshes.append(mesh)
        mesh_to_impl[mesh] = utils.GetMeshImpl(mesh_shape, devices=ds,
                num_nodes=n)

    mtf_shape = utils.ConvertToShape([('axis0', batch_size)] +
        inputs.shape.as_list()[1:])
    mtf_inputs = mtf.import_tf_tensor(meshes[0], inputs, mtf_shape)
    mtf_labels = mtf.import_tf_tensor(meshes[-1], labels, mtf_shape)
    return graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels

class LSTMCell(keras.layers.Layer):
    def __init__(self, num_units, ws, mesh_impl, **kwargs):
        self.num_units = num_units
        self.laid_out_w = ws
        self.mesh_impl = mesh_impl

        _, part_n, part_k = mesh_impl.shape.to_integer_list
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
        assert len(xs) == self.num_gpus
        assert len(states) == 2 * self.num_gpus
        mesh_impl = self.mesh_impl
        devices = mesh_impl.devices
        mesh_dims = mesh_impl.shape.to_integer_list
        axis_n, axis_k = len(mesh_dims)-2, len(mesh_dims)-1

        # State tensors
        hs, cs = states[:self.num_gpus], states[self.num_gpus:]

        xs = [assign_device(x, d) for x, d in zip(xs, devices)]
        hs = [assign_device(h, d) for h, d in zip(hs, devices)]
        cs = [assign_device(c, d) for c, d in zip(cs, devices)]

        # Laid out tensors
        laid_out_x = mesh_impl.LaidOutTensor(list(xs))
        laid_out_h = mesh_impl.LaidOutTensor(list(hs))
        laid_out_c = mesh_impl.LaidOutTensor(list(cs))

        # Concat x and h
        concat_fn = lambda x, y: tf.concat([x, y], axis=1)
        laid_out_xh = mesh_impl.slicewise(concat_fn, laid_out_x, laid_out_h)

        # GEMM: y = xh * w
        laid_out_y = mesh_impl.slicewise(tf.matmul, laid_out_xh, self.laid_out_w)
        if mesh_dims[axis_k] > 1:
            laid_out_y = mesh_impl.allreduce(laid_out_y, [axis_k], "SUM")

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

        # Map last dim of 'hs' from 'axis_n' to 'axis_k'
        if mesh_dims[axis_n] > 1:
            laid_out_h = mesh_impl.allconcat(laid_out_h, axis_n, 1)
        if mesh_dims[axis_k] > 1:
            laid_out_h = mesh_impl.allsplit(laid_out_h, axis_k, 1)

        assert [h.device for h in hs] == devices
        hs = laid_out_h.tensor_list
        cs = laid_out_c.tensor_list
        return hs, hs + cs

class RNNGradOperation(mtf.GenericGradOperation):
    def __init__(self, fwd_op, grad_ys, name=None):
        super().__init__(fwd_op, grad_ys, name)

        assert len(self._outputs) == len(fwd_op.inputs) == (1 + 2 + 4)
        self._outputs[2]._mesh = fwd_op.inputs[2].mesh
        self._outputs[5]._mesh = fwd_op.inputs[5].mesh
        self._outputs[6]._mesh = fwd_op.inputs[6].mesh
        assert all(x.mesh == y.mesh for x, y in zip(fwd_op.inputs,
            self._outputs))

    def lower(self, lowering):
        rnn_op = self._forward_op
        mesh_impls = [lowering.mesh_impl(x) for x in rnn_op.inputs]

        get_tensor_list = lambda x: lowering.tensors[
                x].to_laid_out_tensor().tensor_list

        xs = get_tensor_list(rnn_op.inputs[0])
        ws_l0 = get_tensor_list(rnn_op.inputs[1])
        ws_l1 = get_tensor_list(rnn_op.inputs[2])
        ys = get_tensor_list(rnn_op.outputs[0])
        grad_ys = get_tensor_list(self._grad_ys[0])

        if rnn_op.part_k == rnn_op.part_n == 1:
            # Since we perform RNN as slicewise operation, dy_i/dx_j for i!=j is
            # zero.  So we only compute dy_i/dx_i for various slices.
            # (Replicated) weights are all-reduced separately below.
            assert (len(ys) == len(xs) == len(grad_ys) == len(ws_l0) == len(ws_l1))
            grad_xs_ws = [tf.gradients(y, [x, w0, w1], grad_ys=grad_y,
                colocate_gradients_with_ops=True) for y, x, w0, w1, grad_y in
                zip(ys, xs, ws_l0, ws_l1, grad_ys)]

            assert all(len(g) == 3 for g in grad_xs_ws)
            grad_xs, grad_ws_l0, grad_ws_l1 = utils.TransposeLists(grad_xs_ws)

        else:
            grad_xs_ws = tf.gradients(ys, xs + ws_l0 + ws_l1, grad_ys=grad_ys,
                    colocate_gradients_with_ops=True)
            assert len(grad_xs_ws) == (len(xs) + len(ws_l0) + len(ws_l1))

            grad_xs = grad_xs_ws[:len(xs)]
            grad_ws_l0 = grad_xs_ws[len(xs):len(xs)+len(ws_l0)]
            grad_ws_l1 = grad_xs_ws[len(xs)+len(ws_l0):]

        # Laid out tensors
        grad_xs_lo = mesh_impls[0].LaidOutTensor.from_tensor_list(grad_xs)
        grad_ws_l0_lo = mesh_impls[1].LaidOutTensor.from_tensor_list(grad_ws_l0)
        grad_ws_l1_lo = mesh_impls[2].LaidOutTensor.from_tensor_list(grad_ws_l1)

        # Accumulate dy_i/dw_j for replicated w_j's
        if rnn_op.part_b > 1:
            grad_ws_l0_lo = mesh_impls[1].allreduce(grad_ws_l0_lo, [0], 'SUM')
            grad_ws_l1_lo = mesh_impls[2].allreduce(grad_ws_l1_lo, [0], 'SUM')

        assert len(self.outputs) == (1 + 2 + 4)
        lowering.set_tensor_lowering(self.outputs[0], grad_xs_lo)
        lowering.set_tensor_lowering(self.outputs[1], grad_ws_l0_lo)
        lowering.set_tensor_lowering(self.outputs[2], grad_ws_l1_lo)

class RNNOperation(mtf.Operation):
    def __init__(self, x, w0, w1, states, num_units, name=None):
        assert (x.shape[-1].name == w0.shape[0].name == w1.shape[0].name), (
                x.shape, w0.shape, w1.shape)
        super().__init__([x, w0, w1] + states, mesh=w1.mesh, name=name or 'rnn')
        self.num_units = num_units
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def gradient(self, grad_ys):
        return RNNGradOperation(self, grad_ys, name='rnn_grad').outputs

    def lower(self, lowering):
        mesh_impls = [lowering.mesh_impl(self.inputs[1]),
                lowering.mesh_impl(self.inputs[2])]
        devices = mesh_impls[0].devices + mesh_impls[1].devices
        self.part_b, self.part_n, self.part_k = mesh_impls[0].shape.to_integer_list

        inputs = [lowering.tensors[x] for x in self.inputs]
        x, w0, w1, h0, c0, h1, c1 = mtf.convert_args_to_laid_out_tensors(inputs)
        states = [tuple(h0.tensor_list + c0.tensor_list),
                tuple(h1.tensor_list + c1.tensor_list)]

        # TF device placement selection function
        def device_selector(obj):
            if obj.device:
                return obj.device

            def get_device(name, pattern):
                idx = name.find(pattern)
                if idx == 0:
                    name = name[idx+len(pattern):]
                    idx = name.find(':')
                    if idx >= 0:
                        name = name[:idx]
                        try:
                            idx = int(name)
                        except ValueError:
                            return None
                        return devices[idx]
                return None

            # 'device_selector' is not called for TensorArrays, since keras
            # creates them with default 'colocate_with_first_write_call=True'
            # flag. So, as a workaround, look for op's input tensors with name
            # 'TensorArray_', and set it's op device to be equal to the slice
            # index.
            for t in obj.inputs:
                if not t.device:
                    assert not t.op.device
                    device = get_device(t.name, 'rnn/rnn/TensorArray_')
                    if device:
                        t.op._set_device(device)

            input_devices = set(i.device for i in obj.inputs) - {''}
            if len(input_devices) == 0:
                return ''

            assert len(input_devices) == 1
            return input_devices.pop()

        with tf.device(device_selector):
            cells = [LSTMCell(self.num_units, w0, mesh_impls[0], name='lstm0'),
                    LSTMCell(self.num_units, w1, mesh_impls[1], name='lstm1')]
            tf_rnn_op = keras.layers.RNN(cells, return_sequences=True,
                    return_state=False)
            ys = tf_rnn_op(tuple(x.tensor_list), initial_state=states)

        assert len(ys) == len(mesh_impls[1].devices)
        ys = [assign_device(y, d) for y, d in zip(ys, mesh_impls[1].devices)]
        laid_out_y = mesh_impls[1].LaidOutTensor(ys)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_y)

def model(params, inputs, labels):
    devices = params.devices
    num_gpus = len(devices)

    # MTF mesh
    assert len(inputs.shape) == 2
    graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels = CreateMeshes(
            inputs, labels, params.num_nodes, num_gpus, params.batch_size)
    embed_mesh, lstm0_mesh, lstm1_mesh, proj_mesh = meshes
    batch_dim_name, n_dim_name, k_dim_name = 'axis0', 'axis1', 'axis2'

    # RNN weights
    num_units = params.num_units
    w_shape = utils.ConvertToShape([(k_dim_name, 2*num_units),
        (n_dim_name, 4*num_units)])
    rnn_w0 = mtf.get_variable(lstm0_mesh, 'rnn_w0', w_shape)
    rnn_w1 = mtf.get_variable(lstm1_mesh, 'rnn_w1', w_shape)

    # RNN initial states
    h_shape = mtf.Shape([mtf.Dimension(batch_dim_name, params.batch_size),
        mtf.Dimension(k_dim_name, num_units)])
    c_shape = mtf.Shape([mtf.Dimension(batch_dim_name, params.batch_size),
        mtf.Dimension(n_dim_name, num_units)])
    states0 = [mtf.zeros(lstm0_mesh, h_shape), mtf.zeros(lstm0_mesh, c_shape)]
    states1 = [mtf.zeros(lstm1_mesh, h_shape), mtf.zeros(lstm1_mesh, c_shape)]

    # Model - embedding
    vocab_dim = mtf.Dimension(k_dim_name, params.vocab_size)
    embed_dim = mtf.Dimension(n_dim_name, params.num_units)
    assert mtf_inputs.mesh == embed_mesh
    embedding = mtf.layers.embedding(mtf_inputs, vocab_dim, embed_dim,
            tf.float32)
    assert embedding.shape[-1].name == n_dim_name
    shape = embedding.shape.rename_dimension(n_dim_name, k_dim_name)
    embedding = mesh_trans.ReplaceMeshWithIndependentAxes(
            embedding, lstm0_mesh, shape.dimension_names)

    # Model - RNN
    [y] = RNNOperation(embedding, rnn_w0, rnn_w1, states0 + states1,
            num_units).outputs
    assert y.mesh == lstm1_mesh
    assert y.shape[-1].name == k_dim_name
    assert mesh_to_impl[proj_mesh].shape[-1] == mtf.Dimension(k_dim_name, 1)
    rand_dim_name = utils.RandName()
    y = mtf.rename_dimension(y, k_dim_name, rand_dim_name)
    shape = y.shape.rename_dimension(rand_dim_name, k_dim_name)
    y = mesh_trans.ReplaceMeshWithIndependentAxes(
            y, proj_mesh, shape.dimension_names)

    # Model - Dense + loss
    assert y.shape[-1].name == k_dim_name
    vocab_dim = mtf.Dimension(n_dim_name, params.vocab_size)
    y = mtf.layers.dense(y, vocab_dim, reduced_dims=y.shape[-1:],
            use_bias=False)
    assert mtf_labels.mesh == proj_mesh
    mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(
            y, mtf_labels, vocab_dim)
    mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    model.soft_placement = True
    return graph, mesh_to_impl, mtf_loss
