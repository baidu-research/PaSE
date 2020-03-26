import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import mesh_tensorflow as mtf
import utils
from mesh_transformations import MeshReplacementOperation
from rnnlm import RNNOperation

class ReplaceGNMTMesh(MeshReplacementOperation):
    def lower(self, lowering):
        x = self.inputs[0]
        input_slices = lowering.tensors[x].to_laid_out_tensor().tensor_list
        old_mesh_impl = lowering.mesh_impl(self.old_mesh)
        new_mesh_impl = lowering.mesh_impl(self)
        old_mesh_size = old_mesh_impl.shape.size
        new_mesh_size = new_mesh_impl.shape.size

        if old_mesh_size > new_mesh_size:
            assert old_mesh_size == 2 * new_mesh_size
            assert ((old_mesh_impl.devices[::2] == new_mesh_impl.devices) or
                    (old_mesh_impl.devices[1::2] == new_mesh_impl.devices))

            concat_fn = lambda x, y: tf.concat([x, y], axis=0)
            output_slices = mtf.parallel(new_mesh_impl.devices,
                    concat_fn, input_slices[::2], input_slices[1::2])
        else:
            assert 2 * old_mesh_size == new_mesh_size
            assert ((old_mesh_impl.devices == new_mesh_impl.devices[1::2]) or
                    (old_mesh_impl.devices == new_mesh_impl.devices[::2]))
            assert x.shape[0].size % new_mesh_size == 0
            size = x.shape[0].size // new_mesh_size

            input_slices = utils.FlattenList([(x, x) for x in input_slices])
            slice_fn = lambda x, b: x[b:b+size, ...]
            output_slices = mtf.parallel(new_mesh_impl.devices, slice_fn,
                    input_slices, [0, size] * old_mesh_size)

        laid_out_tensor = new_mesh_impl.LaidOutTensor(output_slices)
        lowering.set_tensor_lowering(self.outputs[0], laid_out_tensor)

def CreateMeshes(inputs, labels, num_nodes, num_gpus, batch_size):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}
    devices = utils.GetDeviceList(num_gpus, num_nodes)

    assert num_gpus % num_nodes == 0
    assert num_gpus % 2 == 0

    mesh = mtf.Mesh(graph, f'mesh0')
    meshes.append(mesh)
    mesh_to_impl[mesh] = utils.GetMeshImpl([num_gpus//2],
            devices=devices[:num_gpus//2])

    mesh = mtf.Mesh(graph, f'mesh1')
    meshes.append(mesh)
    mesh_to_impl[mesh] = utils.GetMeshImpl([num_gpus//2],
            devices=devices[num_gpus//2:])

    mesh = mtf.Mesh(graph, f'mesh2')
    meshes.append(mesh)
    mesh_to_impl[mesh] = utils.GetMeshImpl([num_gpus],
            devices=utils.FlattenList(utils.TransposeLists(
                    [devices[:num_gpus//2], devices[num_gpus//2:]])))

    assert len(inputs.shape) == 2
    assert inputs.shape == labels.shape

    shape = utils.ConvertToShape([('axis0', batch_size),
        inputs.shape.as_list()[1]])
    mtf_inputs = mtf.import_tf_tensor(meshes[2], inputs, shape)
    mtf_labels = mtf.import_tf_tensor(meshes[2], labels, shape)

    return graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels

def model(params, inputs, labels):
    num_gpus = len(params.devices)

    # Mtf mesh
    assert len(inputs.shape) == 2
    graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels = CreateMeshes(
            inputs, labels, params.num_nodes, num_gpus, params.batch_size)

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
    rnn_w1 = mtf.get_variable(meshes[1], 'rnn_w1', w_shape)

    # RNN initial states
    h_shape = mtf.Shape([mtf.Dimension(batch_dim_name, params.batch_size),
        mtf.Dimension(k_dim_name, num_units)])
    c_shape = mtf.Shape([mtf.Dimension(batch_dim_name, params.batch_size),
        mtf.Dimension(n_dim_name, num_units)])
    states0 = [mtf.zeros(meshes[0], h_shape), mtf.zeros(meshes[0], c_shape)]
    states1 = [mtf.zeros(meshes[1], h_shape), mtf.zeros(meshes[1], c_shape)]

    # Model
    embedding = mtf.layers.embedding(mtf_inputs, vocab_dim, embed_dim,
            tf.float32)
    assert embedding.mesh == meshes[2]
    embedding = ReplaceGNMTMesh(embedding, meshes[0]).outputs[0]

    [y] = RNNOperation(embedding, rnn_w0, rnn_w1, num_units,
            states=states0+states1).outputs
    assert y.mesh == meshes[1]
    y = ReplaceGNMTMesh(y, meshes[2]).outputs[0]

    y = mtf.layers.dense(y, vocab_dim, reduced_dims=y.shape[-1:],
            use_bias=False)
    assert y.mesh == mtf_labels.mesh
    mtf_cross_ent = mtf.layers.softmax_cross_entropy_with_logits(
            y, mtf_labels, vocab_dim)
    mtf_loss = mtf.reduce_mean(mtf_cross_ent)

    model.soft_placement = True
    return graph, mesh_to_impl, mtf_loss
