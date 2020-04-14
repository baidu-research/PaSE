import numpy as np
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf

import trainer
import utils
from dataloader import TextDataLoader

class Params():
    def __init__(self, batch_size, vocab_size, max_seq_len, num_nodes, num_gpus):
        self.batch_size = batch_size
        self.num_units = 2048
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = 2
        self.num_nodes = num_nodes
        self.num_gpus = num_gpus

def assign_device(x, d):
    if x.device != d:
        with tf.device(d):
            return tf.identity(x)
    else:
        return x

class LSTMCell(tf.keras.layers.Layer):
    def __init__(self, num_units, ws, mesh_impl, mesh_axis_n, mesh_axis_k,
            **kwargs):
        self.num_units = num_units
        self.laid_out_w = ws
        self.mesh_impl = mesh_impl
        self.num_gpus = mesh_impl.size

        get_axis_info = lambda axis: ((axis.size, mesh_impl.shape.dims.index(
            axis)) if axis is not None else (1, None))
        self.part_n, self.axis_n = get_axis_info(mesh_axis_n)
        self.part_k, self.axis_k = get_axis_info(mesh_axis_k)

        assert num_units % self.part_k == 0
        assert num_units % self.part_n == 0
        part_k_size = num_units // self.part_k
        part_n_size = num_units // self.part_n

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
        if self.part_k > 1:
            laid_out_y = mesh_impl.allreduce(laid_out_y, [self.axis_k], "SUM")

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
        # There is no need for any commnunication for elementwise operations
        # even if the n-dimension of 'laid_out_y' is distributed, since
        # permuting the columns of weight matrix does not semantically change
        # the computation. So, for eg when num_gpus=2 and n-dim of laid_out_y is
        # distributed, we assume that rather than the columns of 'laid_out_y'
        # being 'i|f||g|o' distributed among 2 gpus, they are split as follows:
        # 'i1|f1|g1|o1||i2|f2|g2|o2', so the corresponding columns of i,f,g,o
        # are owned by the same gpu without shuffling.
        laid_out_h, laid_out_c = mesh_impl.slicewise(act_fn, laid_out_y,
                laid_out_c)

        # Map last dim of 'hs' from 'axis_n' to 'axis_k'
        if self.part_n > 1:
            laid_out_h = mesh_impl.allconcat(laid_out_h, self.axis_n, 1)
        if self.part_k > 1:
            laid_out_h = mesh_impl.allsplit(laid_out_h, self.axis_k, 1)

        assert [h.device for h in hs] == devices
        hs = laid_out_h.tensor_list
        cs = laid_out_c.tensor_list
        return hs, hs + cs

class RNNGradOperation(mtf.GenericGradOperation):
    def __init__(self, fwd_op, grad_ys, name=None):
        super().__init__(fwd_op, grad_ys, name)
        assert ((fwd_op.inputs[0].mesh == self._outputs[0].mesh) and
                (fwd_op.inputs[1].mesh == self._outputs[1].mesh))
        for x, y in zip(self._outputs[2:], fwd_op.inputs[2:]):
            x._mesh = y.mesh

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

        get_axis_size = lambda ma: ma.size if ma is not None else 1
        part_b = get_axis_size(rnn_op.mesh_axis_b)
        part_k = get_axis_size(rnn_op.mesh_axis_k)
        part_n = get_axis_size(rnn_op.mesh_axis_n)

        if part_k == part_n == 1:
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
        if part_b > 1:
            grad_ws_l0_lo = mesh_impls[1].allreduce(grad_ws_l0_lo, [0], 'SUM')
            grad_ws_l1_lo = mesh_impls[2].allreduce(grad_ws_l1_lo, [0], 'SUM')

        lowering.set_tensor_lowering(self.outputs[0], grad_xs_lo)
        lowering.set_tensor_lowering(self.outputs[1], grad_ws_l0_lo)
        lowering.set_tensor_lowering(self.outputs[2], grad_ws_l1_lo)

class RNNOperation(mtf.Operation):
    def __init__(self, x, w0, w1, num_units, states, name=None):
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

        def get_mesh_axis(x, axis):
            mesh_impl = lowering.mesh_impl(x)
            ma = mesh_impl.tensor_layout(x).tensor_axis_to_mesh_axis[axis]
            return mesh_impl.shape[ma] if ma is not None else None
        self.mesh_axis_b = get_mesh_axis(self.inputs[0], 0)
        self.mesh_axis_k = get_mesh_axis(self.inputs[1], 0)
        self.mesh_axis_n = get_mesh_axis(self.inputs[1], 1)

        inputs = [lowering.tensors[x] for x in self.inputs]
        x, w0, w1, h0, c0, h1, c1 = mtf.convert_args_to_laid_out_tensors(inputs)
        states = [tuple(h0.tensor_list + c0.tensor_list),
                tuple(h1.tensor_list + c1.tensor_list)]

        # TF device placement selection function
        def device_selector(obj):
            if obj.device:
                return obj.device

            def get_device(name):
                pattern = 'rnn/rnn/TensorArray'
                assert name.startswith(pattern)
                name = name[len(pattern):]
                try:
                    idx = int(name[1:])
                    return devices[idx]
                except (IndexError, ValueError):
                    return devices[0]

            # 'device_selector' is not called for TensorArrays, since keras
            # creates them with default 'colocate_with_first_write_call=True'
            # flag. So, as a workaround, look for op's input tensors with type
            # 'TensorArrayV3', and set it's op device to be equal to the slice
            # index.
            for t in obj.inputs:
                op = t.op
                if (op.type == 'TensorArrayV3') and (not op.device):
                    op._set_device(get_device(op.name))

            input_devices = set(i.device for i in obj.inputs) - {''}
            if len(input_devices) == 0:
                return ''

            assert len(input_devices) == 1
            return input_devices.pop()

        with tf.device(device_selector):
            cells = [LSTMCell(self.num_units, w0, mesh_impls[0],
                self.mesh_axis_n, self.mesh_axis_k),
                    LSTMCell(self.num_units, w1, mesh_impls[1],
                        self.mesh_axis_n, self.mesh_axis_k)]
            tf_rnn_op = tf.keras.layers.RNN(cells, return_sequences=True,
                    return_state=False)
            ys = tf_rnn_op(tuple(x.tensor_list), initial_state=states)

        assert len(ys) == len(mesh_impls[1].devices)
        ys = [assign_device(y, d) for y, d in zip(ys, mesh_impls[1].devices)]
        laid_out_y = mesh_impls[1].LaidOutTensor(ys)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_y)

def main():
    t = trainer.Trainer()
    args = t.args
    lr = 0.01

    # Initialize dataset
    dataset = TextDataLoader(args.batch_size, args.src_vocab, None,
            args.src_text, None, args.seq_len, args.src_vocab_size,
            args.tgt_vocab_size, args.sentences_size)
    inputs, labels, _, _ = dataset.next_batch()

    # Convert inputs and labels to int32, due to a bug in mtf.one_hot that leads
    # to TypeError due to type mismatch
    inputs = tf.cast(inputs, tf.int32)
    labels = tf.cast(labels, tf.int32)

    vocab_size = utils.RoundUp(dataset.src_vocab_size, t.num_gpus)
    print("Vocab size: %d" % vocab_size)
    params = Params(args.batch_size, vocab_size, args.seq_len,
            t.num_nodes, t.num_gpus)

    # Model
    if args.strategy == 0:
        import rnnlm_data as rnn
    elif args.strategy == 1:
        import rnnlm_opt as rnn
    elif args.strategy == 2:
        import rnnlm_gnmt as rnn
    elif args.strategy == 3:
        import rnnlm_flexflow as rnn
    else:
        assert False
    graph, mesh_to_impl, mtf_loss = rnn.model(params, inputs, labels)

    #try:
    #    soft_placement = rnn.model.soft_placement
    #except AttributeError:
    #    soft_placement = False
    soft_placement = True

    # Train
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    config = tf.ConfigProto(allow_soft_placement=soft_placement,
            log_device_placement=True)
    t.train_model(graph, mesh_to_impl, mtf_loss, dataset, config=config,
            run_options=run_options)


if __name__ == '__main__':
    main()

