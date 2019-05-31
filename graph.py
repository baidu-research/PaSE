import networkx as nx

import nn_ops


def AlexNet(b):
    img = nn_ops.Tensor((b, 3, 227, 227))
    img.SetAsInput()

    # Conv1 + relu + maxpool
    conv1 = nn_ops.Conv(img, (96, 3, 11, 11), stride=4, pw_op_cnt=1)
    pool1 = nn_ops.Pooling(conv1.GetOutTensor(0), (3, 3), stride=2)

    # Conv2 + relu + maxpool
    conv2 = nn_ops.Conv(pool1.GetOutTensor(0), (256, 96, 5, 5), pad=2,
            pw_op_cnt=1)
    pool2 = nn_ops.Pooling(conv2.GetOutTensor(0), (3, 3), stride=2)

    # Conv3 + relu
    conv3 = nn_ops.Conv(pool2.GetOutTensor(0), (384, 256, 3, 3), stride=1,
            pad=1, pw_op_cnt=1)

    # Conv4 + relu
    conv4 = nn_ops.Conv(conv3.GetOutTensor(0), (384, 384, 3, 3), stride=1,
            pad=1, pw_op_cnt=1)

    # Conv5 + relu + maxpool
    conv5 = nn_ops.Conv(conv4.GetOutTensor(0), (256, 384, 3, 3), stride=1,
            pad=1, pw_op_cnt=1)
    pool5 = nn_ops.Pooling(conv5.GetOutTensor(0), (3, 3), stride=2)

    # Reshape
    reshape = nn_ops.Reshape(pool5.GetOutTensor(0), (b, 256 * 6 * 6))

    # FC6 + relu
    fc6 = nn_ops.FC(reshape.GetOutTensor(0), 4096, pw_op_cnt=1)

    # FC7 + relu
    fc7 = nn_ops.FC(fc6.GetOutTensor(0), 4096, pw_op_cnt=1)

    # FC8
    fc8 = nn_ops.FC(fc7.GetOutTensor(0), 1024)

    # Softmax + cross-entropy loss
    loss = nn_ops.SoftmaxCrossEntropy(fc8.GetOutTensor(0))

    return nn_ops.Ops.G


'''
def ResNet101(G, b, n_procs):
    img = nn_ops.Tensor((b, 3, 227, 227))
    blocks = (3, 4, 23, 3)
    planes = (64, 128, 256, 512)
    num_classes = 1000
    expansion = 4

    # Conv1 + bn + relu + maxpool
    conv1 = nn_ops.Conv(img, (64, 3, 7, 7), stride=2, pad=3)
    node_id1_0 = AddVertex(G, conv1)
    bn1 = nn_ops.BatchNorm(conv1.GetOutTensor(0))
    node_id1_1 = AddVertex(G, bn1)
    AddEdge(G, node_id1_0, node_id1_1)
    pool1 = nn_ops.Pooling(bn1.GetOutTensor(0), (3, 3)


    return G


def InceptionV3(G, b):
    img = nn_ops.Tensor((b, 3, 299, 299))
    num_classes = 1000
'''


# Creates the graph for the model
def CreateGraph(graph_type, batch_size, hidden_dim_size, n_procs):
    nn_ops.Ops.default_procs = n_procs

    if graph_type == 'alexnet':
        G = AlexNet(batch_size)
    elif graph_type == 'resnet101':
        G = ResNet101(batch_size, n_procs).Graph()
    elif graph_type == 'inception3':
        G = Inception3(batch_size, n_procs).Graph()
    elif graph_type == 'seq2seq':
        G = Seq2seq(batch_size, n_procs)
    else:
        assert(False)

    return G

