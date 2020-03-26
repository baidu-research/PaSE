import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import string
import utils
import mesh_transformations as mesh_trans
import mtf_operations as mt
from rnnlm import RNNOperation

def CreateMeshes(inputs, labels, num_nodes, num_gpus, batch_size):
    graph = mtf.Graph()
    meshes = []
    mesh_to_impl = {}
    devices = utils.GetDeviceList(num_gpus, num_nodes)

    assert len(inputs.shape) == 2
    assert inputs.shape == labels.shape

    if num_gpus == 4:
        # Mesh_shape: batch_dim, n_dim, k_dim
        mesh_shapes = [[1, 1, 4],
                       [2, 1, 1],
                       [2, 1, 1],
                       [1, 4, 1]]

    elif num_gpus == 8:
        # Mesh_shape: batch_dim, n_dim, k_dim
        mesh_shapes = [[1, 1, 8],
                       [4, 1, 1],
                       [4, 1, 1],
                       [1, 8, 1]]

    elif num_gpus == 16:
        # Mesh_shape: batch_dim, n_dim, k_dim
        mesh_shapes = [[1, 1, 16],
                       [2, 2, 2],
                       [2, 2, 2],
                       [1, 16, 1]]

    elif num_gpus == 32:
        # Mesh_shape: batch_dim, n_dim, k_dim
        mesh_shapes = [[1, 1, 32],
                       [4, 2, 2],
                       [4, 2, 2],
                       [1, 32, 1]]

    elif num_gpus == 64:
        # Mesh_shape: batch_dim, n_dim, k_dim
        mesh_shapes = [[1, 1, 64],
                       [8, 2, 2],
                       [8, 2, 2],
                       [1, 64, 1]]

    else:
        assert False

    assert mesh_shapes[1] == mesh_shapes[2]
    assert (utils.Prod(mesh_shapes[1]) == utils.Prod(mesh_shapes[2]) ==
            num_gpus // 2)
    assert (num_nodes == 1) or (num_nodes % 2 == 0)
    half_devices0 = devices[:(num_gpus // 2)]
    half_devices1 = devices[(num_gpus // 2):]
    mesh_devices = [devices, half_devices0, half_devices1,
            half_devices1 + half_devices0]
    node_counts = [num_nodes, max(1, num_nodes // 2),
            max(1, num_nodes // 2), num_nodes]

    for i, (mesh_shape, ds, n) in enumerate(
            zip(mesh_shapes, mesh_devices, node_counts)):
        mesh = mtf.Mesh(graph, 'mesh' + str(i))
        meshes.append(mesh)
        mesh_to_impl[mesh] = utils.GetMeshImpl(mesh_shape, devices=ds,
                num_nodes=n)

    mtf_shape = utils.ConvertToShape([('axis0', batch_size)] +
        inputs.shape.as_list()[1:])
    mtf_inputs = mtf.import_tf_tensor(meshes[0], inputs, mtf_shape)
    mtf_labels = mtf.import_tf_tensor(meshes[-1], labels, mtf_shape)
    return graph, meshes, mesh_to_impl, mtf_inputs, mtf_labels

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
    [y] = RNNOperation(embedding, rnn_w0, rnn_w1, num_units,
            states=states0 + states1).outputs
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
