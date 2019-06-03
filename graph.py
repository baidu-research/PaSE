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


def ResNet101(b, n_procs):
    img = nn_ops.Tensor((b, 3, 227, 227))
    img.SetAsInput()
    layers = (3, 4, 23, 3)

    num_classes = 1000
    expansion = 4
    inplanes = 64

    def Bottleneck(img, inplanes, planes, stride=1, downsample=None):
        identity = img

        conv1 = nn_ops.Conv(img, (planes, inplanes, 1, 1))
        bn1 = nn_ops.BatchNorm(conv1.GetOutTensor(0))

        conv2 = nn_ops.Conv(bn1.GetOutTensor(0), (planes, planes, 3, 3),
                stride=stride, pad=1)
        bn2 = nn_ops.BatchNorm(conv2.GetOutTensor(0))

        conv3 = nn_ops.Conv(bn2.GetOutTensor(0), (planes * expansion, planes, 1,
            1))
        bn3 = nn_ops.BatchNorm(conv3.GetOutTensor(0))

        if downsample is not None:
            identity = downsample(img)

        out = nn_ops.Elementwise(bn3.GetOutTensor(0), identity, pw_op_cnt=1)
        return out

    def MakeLayer(img, planes, blocks, stride=1):
        downsample = None
        nonlocal inplanes

        if stride != 1 or inplanes != planes * expansion:
            downsample = lambda x: nn_ops.BatchNorm(nn_ops.Conv(x, (planes *
                expansion, inplanes, 1, 1),
                stride=stride).GetOutTensor(0)).GetOutTensor(0)

        layers = Bottleneck(img, inplanes, planes, stride, downsample)
        inplanes = planes * expansion
        for _ in range(1, blocks):
            layers = Bottleneck(layers.GetOutTensor(0), inplanes, planes)

        return layers

    # Conv1 + bn + relu + maxpool
    conv1 = nn_ops.Conv(img, (64, 3, 7, 7), stride=2, pad=3)
    bn1 = nn_ops.BatchNorm(conv1.GetOutTensor(0))
    pool1 = nn_ops.Pooling(bn1.GetOutTensor(0), (3, 3), stride=2, pad=1)

    # Layers
    layer1 = MakeLayer(pool1.GetOutTensor(0), 64, layers[0])
    layer2 = MakeLayer(layer1.GetOutTensor(0), 128, layers[1], stride=2)
    layer3 = MakeLayer(layer2.GetOutTensor(0), 256, layers[2], stride=2)
    layer4 = MakeLayer(layer3.GetOutTensor(0), 512, layers[3], stride=2)

    # Avg pooling + FC
    mean = nn_ops.ReduceMean(layer4.GetOutTensor(0), axis=[2,3], keepdims=True)
    tsr = mean.GetOutTensor(0)
    assert len(tsr) == 4
    reshape = nn_ops.Reshape(tsr, (tsr[0], tsr[1] * tsr[2] * tsr[3]))
    fc = nn_ops.FC(reshape.GetOutTensor(0), num_classes)

    return nn_ops.Ops.G


'''
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
        G = ResNet101(batch_size, n_procs)
    elif graph_type == 'inception3':
        G = Inception3(batch_size, n_procs).Graph()
    elif graph_type == 'seq2seq':
        G = Seq2seq(batch_size, n_procs)
    else:
        assert(False)

    return G

