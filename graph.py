import networkx as nx
import functools

import nn_ops


def AlexNet(b):
    img = nn_ops.InputTensor((b, 3, 227, 227))

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


def ResNet101(b):
    img = nn_ops.InputTensor((b, 3, 227, 227))
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
    mean = nn_ops.ReduceMean(layer4.GetOutTensor(0), axis=[2,3])
    fc = nn_ops.FC(mean.GetOutTensor(0), num_classes)

    # Softmax + cross-entropy loss
    loss = nn_ops.SoftmaxCrossEntropy(fc.GetOutTensor(0))

    return nn_ops.Ops.G


def Inception3(b, aux_logits=False):
    img = nn_ops.InputTensor((b, 3, 299, 299))
    num_classes = 1000

    def AddBasicConv(img, fltr, stride=1, padding=0):
        conv = nn_ops.Conv(img, fltr, stride, padding)
        bn = nn_ops.BatchNorm(conv.GetOutTensor(0))
        return bn

    def AddInceptionA(img, in_channels, pool_features):
        branch1x1 = AddBasicConv(img, (64, in_channels, 1, 1))

        branch5x5 = AddBasicConv(img, (48, in_channels, 1, 1))
        branch5x5 = AddBasicConv(branch5x5.GetOutTensor(0), (64, 48, 5, 5),
                padding=2)

        branch3x3dbl = AddBasicConv(img, (64, in_channels, 1, 1))
        branch3x3dbl = AddBasicConv(branch3x3dbl.GetOutTensor(0), (96, 64, 3,
            3), padding=1)
        branch3x3dbl = AddBasicConv(branch3x3dbl.GetOutTensor(0), (96, 96, 3,
            3), padding=1)

        branch_pool = nn_ops.Pooling(img, (3, 3), stride=1, pad=1)
        branch_pool = AddBasicConv(branch_pool.GetOutTensor(0), (pool_features,
            in_channels, 1, 1))

        outputs = nn_ops.Concat([branch1x1.GetOutTensor(0),
            branch5x5.GetOutTensor(0), branch3x3dbl.GetOutTensor(0),
            branch_pool.GetOutTensor(0)], 1)
        return outputs

    def AddInceptionB(img, in_channels):
        branch3x3 = AddBasicConv(img, (384, in_channels, 3, 3), stride=2)

        branch3x3dbl = AddBasicConv(img, (64, in_channels, 1, 1))
        branch3x3dbl = AddBasicConv(branch3x3dbl.GetOutTensor(0), (96, 64, 3, 3), padding=1)
        branch3x3dbl = AddBasicConv(branch3x3dbl.GetOutTensor(0), (96, 96, 3, 3), stride=2)

        branch_pool = nn_ops.Pooling(img, (3, 3), stride=2)

        outputs = nn_ops.Concat([branch3x3.GetOutTensor(0),
            branch3x3dbl.GetOutTensor(0), branch_pool.GetOutTensor(0)], 1)
        return outputs

    def AddInceptionC(img, in_channels, channels_7x7):
        branch1x1 = AddBasicConv(img, (192, in_channels, 1, 1))

        branch7x7 = AddBasicConv(img, (channels_7x7, in_channels, 1, 1))
        branch7x7 = AddBasicConv(branch7x7.GetOutTensor(0), (channels_7x7,
            channels_7x7, 1, 7), padding=(0, 3))
        branch7x7 = AddBasicConv(branch7x7.GetOutTensor(0), (192, channels_7x7,
            7, 1), padding=(3, 0))

        branch7x7_dbl = AddBasicConv(img, (channels_7x7, in_channels, 1, 1))
        branch7x7_dbl = AddBasicConv(branch7x7_dbl.GetOutTensor(0),
                (channels_7x7, channels_7x7, 7, 1), padding=(3, 0))
        branch7x7_dbl = AddBasicConv(branch7x7_dbl.GetOutTensor(0),
                (channels_7x7, channels_7x7, 1, 7), padding=(0, 3))
        branch7x7_dbl = AddBasicConv(branch7x7_dbl.GetOutTensor(0),
                (channels_7x7, channels_7x7, 7, 1), padding=(3, 0))
        branch7x7_dbl = AddBasicConv(branch7x7_dbl.GetOutTensor(0), (192,
            channels_7x7, 1, 7), padding=(0, 3))

        branch_pool = nn_ops.Pooling(img, (3, 3), stride=1, pad=1)
        branch_pool = AddBasicConv(branch_pool.GetOutTensor(0), (192,
            in_channels, 1, 1))

        outputs = nn_ops.Concat([branch1x1.GetOutTensor(0),
            branch7x7.GetOutTensor(0), branch7x7_dbl.GetOutTensor(0),
            branch_pool.GetOutTensor(0)], 1)
        return outputs

    def AddInceptionD(img, in_channels):
        branch3x3 = AddBasicConv(img, (192, in_channels, 1, 1))
        branch3x3 = AddBasicConv(branch3x3.GetOutTensor(0), (320, 192, 3, 3), stride=2)

        branch7x7x3 = AddBasicConv(img, (192, in_channels, 1, 1))
        branch7x7x3 = AddBasicConv(branch7x7x3.GetOutTensor(0), (192, 192, 1,
            7), padding=(0, 3))
        branch7x7x3 = AddBasicConv(branch7x7x3.GetOutTensor(0), (192, 192, 7,
            1), padding=(3, 0))
        branch7x7x3 = AddBasicConv(branch7x7x3.GetOutTensor(0), (192, 192, 3,
            3), stride=2)

        branch_pool = nn_ops.Pooling(img, (3, 3), stride=2)
        outputs = nn_ops.Concat([branch3x3.GetOutTensor(0),
            branch7x7x3.GetOutTensor(0), branch_pool.GetOutTensor(0)], 1)
        return outputs

    def AddInceptionE(img, in_channels):
        branch1x1 = AddBasicConv(img, (320, in_channels, 1, 1))

        branch3x3 = AddBasicConv(img, (384, in_channels, 1, 1))
        branch3x3_2a = AddBasicConv(branch3x3.GetOutTensor(0), (384, 384, 1, 3),
                padding=(0, 1))
        branch3x3_2b = AddBasicConv(branch3x3.GetOutTensor(0), (384, 384, 3, 1),
                padding=(1, 0))
        branch3x3 = nn_ops.Concat([branch3x3_2a.GetOutTensor(0),
            branch3x3_2b.GetOutTensor(0)], 1)

        branch3x3dbl = AddBasicConv(img, (448, in_channels, 1, 1))
        branch3x3dbl = AddBasicConv(branch3x3dbl.GetOutTensor(0), (384, 448, 3,
            3), padding=1)
        branch3x3dbl_3a = AddBasicConv(branch3x3dbl.GetOutTensor(0), (384, 384,
            1, 3), padding=(0, 1))
        branch3x3dbl_3b = AddBasicConv(branch3x3dbl.GetOutTensor(0), (384, 384,
            3, 1), padding=(1, 0))
        branch3x3dbl = nn_ops.Concat([branch3x3dbl_3a.GetOutTensor(0),
            branch3x3dbl_3b.GetOutTensor(0)], 1)

        branch_pool = nn_ops.Pooling(img, (3, 3), stride=1, pad=1)
        branch_pool = AddBasicConv(branch_pool.GetOutTensor(0), (192,
            in_channels, 1, 1))

        outputs = nn_ops.Concat([branch1x1.GetOutTensor(0),
            branch3x3.GetOutTensor(0), branch3x3dbl.GetOutTensor(0),
            branch_pool.GetOutTensor(0)], 1)
        return outputs

    def AddInceptionAux(img, in_channels, num_classes):
        pool = nn_ops.Pooling(img, (5, 5), stride=3)
        conv0 = AddBasicConv(pool.GetOutTensor(0), (128, in_channels, 1, 1))
        conv1 = AddBasicConv(conv0.GetOutTensor(0), (768, 128, 5, 5))
        mean = nn_ops.ReduceMean(conv1.GetOutTensor(0), axis=[2,3], keepdims=True)
        fc = nn_ops.FC(mean.GetOutTensor(0), num_classes)
        return fc

    conv1a = AddBasicConv(img, (32, 3, 3, 3), stride=2)
    conv2a = AddBasicConv(conv1a.GetOutTensor(0), (32, 32, 3, 3))
    conv2b = AddBasicConv(conv2a.GetOutTensor(0), (64, 32, 3, 3), padding=1)
    pool = nn_ops.Pooling(conv2b.GetOutTensor(0), (3, 3), stride=2)

    conv3b =AddBasicConv(pool.GetOutTensor(0), (80, 64, 1, 1))
    conv4a = AddBasicConv(conv3b.GetOutTensor(0), (192, 80, 3, 3))
    pool = nn_ops.Pooling(conv4a.GetOutTensor(0), (3, 3), stride=2)

    mixed5b = AddInceptionA(pool.GetOutTensor(0), 192, 32)
    mixed5c = AddInceptionA(mixed5b.GetOutTensor(0), 256, 64)
    mixed5d = AddInceptionA(mixed5c.GetOutTensor(0), 288, 64)
    mixed6a = AddInceptionB(mixed5d.GetOutTensor(0), 288)
    mixed6b = AddInceptionC(mixed6a.GetOutTensor(0), 768, 128)
    mixed6c = AddInceptionC(mixed6b.GetOutTensor(0), 768, 160)
    mixed6d = AddInceptionC(mixed6c.GetOutTensor(0), 768, 160)
    mixed6e = AddInceptionC(mixed6d.GetOutTensor(0), 768, 192)
    #if aux_logits:
    #    aux = InceptionAux(mixed6e.GetOutTensor(0), 768, num_classes)
    mixed7a = AddInceptionD(mixed6e.GetOutTensor(0), 768)
    mixed7b = AddInceptionE(mixed7a.GetOutTensor(0), 1280)
    mixed7c = AddInceptionE(mixed7b.GetOutTensor(0), 2048)

    mean = nn_ops.ReduceMean(mixed7c.GetOutTensor(0), axis=[2,3])
    fc = nn_ops.FC(mean.GetOutTensor(0), num_classes)

    # Softmax + cross-entropy loss
    loss = nn_ops.SoftmaxCrossEntropy(fc.GetOutTensor(0))

    return nn_ops.Ops.G


def Transformer(b):
    max_seq_len = 256
    vocab_size = 50000
    embed_dim = 1024
    heads = 8
    ff_dim = heads * 512
    nx = 6
    d_k = 128

    enc_inp_tsr = nn_ops.InputTensor((b, max_seq_len))
    dec_inp_tsr = nn_ops.InputTensor((b, max_seq_len))
    pos_enc = nn_ops.InputTensor((b, max_seq_len, embed_dim))

    # Multi-head attention layer
    def MultiheadAttention(q, k, v):
        '''
        # Multi-head
        k = nn_ops.FC(k, embed_dim)
        k = nn_ops.Reshape(k.GetOutTensor(0), (b, max_seq_len, heads, int(embed_dim
            / heads)))
        k = nn_ops.Transpose(k.GetOutTensor(0), (0, 2, 1, 3))

        v = nn_ops.FC(v, embed_dim)
        v = nn_ops.Reshape(v.GetOutTensor(0), (b, max_seq_len, heads, int(embed_dim
            / heads)))
        v = nn_ops.Transpose(v.GetOutTensor(0), (0, 2, 1, 3))

        q = nn_ops.FC(q, embed_dim)
        q = nn_ops.Reshape(q.GetOutTensor(0), (b, max_seq_len, heads, int(embed_dim
            / heads)))
        q = nn_ops.Transpose(q.GetOutTensor(0), (0, 2, 1, 3))
 
        # Attention
        k_t = nn_ops.Transpose(k, (0, 1, 3, 2))(0)
        qk = nn_ops.MatMul(q, k_t)(0)
        smax = nn_ops.Softmax(qk, axis=3)(0)
        scores = nn_ops.MatMul(smax, v)(0)

        # Multi-head final linear layer
        scores = nn_ops.Transpose(scores, (0, 2, 1, 3))(0)
        scores = nn_ops.Reshape(scores, (b * max_seq_len, embed_dim))(0)
        scores = nn_ops.FC(scores, embed_dim)
        '''

        # Multihead
        # s: stack, b: batch, l: seq_len, e: embed_dim, h: heads, k: d_k
        eq = 'sble,shek->sbhlk'
        qkv = nn_ops.Stack((q, k, v))(0)
        wqkv = nn_ops.InputTensor((3, heads, embed_dim, d_k))
        qkv = nn_ops.Einsum(eq, qkv, wqkv, trainable=True)(0)
        q, k, v = nn_ops.Unstack(qkv)()

        # Dot-product attention
        eq = 'bhlk,bhmk->bhlm' # Memory length: m = l
        logits = nn_ops.Einsum(eq, q, k, trainable=False)(0)
        weights = nn_ops.Softmax(logits, axis=3)(0)
        eq = 'bhlm,bhmk->bhlk'
        scores = nn_ops.Einsum(eq, weights, v, trainable=False)(0)

        # Final linear layer
        wo = nn_ops.InputTensor((heads, embed_dim, d_k))
        eq = 'bhlk,hek->ble'
        return nn_ops.Einsum(eq, scores, wo, trainable=True)

    # Feed-forward network: FF + relu + dropout + FF
    def FeedFwd(inp_tsr):
        eq = 'ble,ef->blf'
        w = nn_ops.InputTensor((embed_dim, ff_dim))
        ff = nn_ops.Einsum(eq, inp_tsr, w, trainable=True, pw_op_cnt=2)(0)

        eq = 'blf,fe->ble'
        w = nn_ops.InputTensor((ff_dim, embed_dim))
        return nn_ops.Einsum(eq, ff, w, trainable=True)

    # Encoder layer
    def Encoder(inp_tsr):
        norm1 = nn_ops.Norm(inp_tsr)(0)
        att = MultiheadAttention(norm1, norm1, norm1)(0)
        att = nn_ops.Elementwise(inp_tsr, att)(0)

        norm2 = nn_ops.Norm(att)(0)
        ff = FeedFwd(norm2)(0)
        return nn_ops.Elementwise(att, ff)

    # Decoder layer
    def Decoder(inp_tsr, enc_out_tsr):
        norm1 = nn_ops.Norm(inp_tsr)(0)
        att1 = MultiheadAttention(norm1, norm1, norm1)(0)
        att1 = nn_ops.Elementwise(inp_tsr, att1)(0)

        norm2 = nn_ops.Norm(att1)(0)
        att2 = MultiheadAttention(norm2, enc_out_tsr, enc_out_tsr)(0)
        att2 = nn_ops.Elementwise(att1, att2)(0)

        norm3 = nn_ops.Norm(att2)(0)
        ff = FeedFwd(norm3)(0)
        return nn_ops.Elementwise(att2, ff)

    # Encoder
    embed = nn_ops.Embedding(enc_inp_tsr, vocab_size, embed_dim)
    pe = nn_ops.Elementwise(embed.GetOutTensor(0), pos_enc)
    x = pe
    for _ in range(nx):
        x = Encoder(x.GetOutTensor(0))
    enc = nn_ops.Norm(x.GetOutTensor(0))

    # Decoder
    embed = nn_ops.Embedding(dec_inp_tsr, vocab_size, embed_dim)
    pe = nn_ops.Elementwise(embed.GetOutTensor(0), pos_enc)
    x = pe
    for _ in range(nx):
        x = Decoder(x.GetOutTensor(0), enc.GetOutTensor(0))
    dec = nn_ops.Norm(x.GetOutTensor(0))(0)

    # Linear + Softmax + cross-entropy loss
    eq = 'ble,ev->blv'
    w = nn_ops.InputTensor((embed_dim, vocab_size))
    dec = nn_ops.Einsum(eq, dec, w, trainable=True)(0)
    loss = nn_ops.SoftmaxCrossEntropy(dec)
    return nn_ops.Ops.G

'''
def Seq2seq(b):
    num_layers = 2
    vocab_size = 50000
    embed_dim = 1024
    max_seq_len = 64
    unroll_factor = 1
    assert max_seq_len % unroll_factor == 0
    repeat_steps = int(max_seq_len / unroll_factor)

    def RNNLoop(hidden=None, context=None, attention_fn=None,
            entry_nodes=[]):
        hidden = hidden or [None] * num_layers
        assert len(hidden) == num_layers

        inp = nn_ops.InputTensor((b,))
        ops = []
        outputs = []
        for i in range(unroll_factor):
            # Embedding
            cell_input_op = nn_ops.Embedding(inp, vocab_size, embed_dim)
            ops.append(cell_input_op)

            # RNN layers
            stack = []
            next_context = None
            for j in range(num_layers):
                cell_input_op = nn_ops.LSTMCell(cell_input_op(0), embed_dim,
                        hidden[j], context=context)
                stack.append(cell_input_op)

                if j == 0 and (attention_fn is not None):
                    attention_ops = attention_fn(cell_input_op(0))
                    ops += attention_ops
                    next_context = attention_ops[-1](0)
            context = next_context
            ops += stack

            outputs.append(stack[-1](0))
            hidden = [s(0) for s in stack]
            if i == 0:
                first_stack = stack

        # Create back edges
        for op1, op2 in zip(first_stack, stack):
            op2.AddEdge(op1(0), 0)
        try: # If we computed attention, create edges from it
            attention = attention_ops[-1](0)
            [op.AddEdge(attention, 0) for op in first_stack]
        except UnboundLocalError:
            pass

        # Repeat RNN 'repeat_steps' times
        entry_edges = [e for e in nn_ops.Ops.G.out_edges(entry_nodes)]
        nn_ops.Repeat(ops, repeat_steps, entry_edges)

        # Repeat outputs 'repeat_steps' times
        out = nn_ops.Stack(outputs, axis=1)
        out = nn_ops.Extend(out(0), repeat_steps, axis=1)

        return out, hidden

    # Encoder
    enc_out, hidden = RNNLoop()

    # Attention
    attn_w = nn_ops.InputTensor((embed_dim, embed_dim))
    attn_dense = nn_ops.Einsum('ble,ef->blf', enc_out(0), attn_w,
            trainable=True)(0)

    def AttnFn(x):
        weights = nn_ops.Einsum('ble,be->bl', attn_dense, x)
        score = nn_ops.Softmax(weights(0), axis=1)
        context = nn_ops.Einsum('bl,ble->be', score(0), enc_out(0),
                trainable=False)
        return [weights, score, context]

    # Decoder
    context = nn_ops.InputTensor((b, embed_dim))
    dec_out, _ = RNNLoop(hidden, context, AttnFn, [attn_dense])

    # Final projection + Loss
    w = nn_ops.InputTensor((embed_dim, vocab_size))
    proj = nn_ops.Einsum('ble,ev->blv', dec_out(0), w, trainable=True)(0)
    loss = nn_ops.SoftmaxCrossEntropy(proj)
    return nn_ops.Ops.G
'''


def RNNLM(b):
    num_layers = 2
    vocab_size = 50000
    num_units = 2048
    max_seq_len = 64
    unroll_factor = 8

    inp_tsr = nn_ops.InputTensor((b, max_seq_len))
    embed = nn_ops.Embedding(inp_tsr, vocab_size, num_units)(0)
    xs = nn_ops.Unstack(embed, axis=1)()
    w = nn_ops.Variable(nn_ops.InputTensor((2*num_units, 4*num_units)))(0)
    h = None

    ys = []
    for i in range(unroll_factor):
        y = nn_ops.LSTMCell(xs[i], num_units, w, h)(0)
        ys.append(y)
        h = y

    ys = nn_ops.Stack(ys, axis=1)(0)
    loss = nn_ops.SoftmaxCrossEntropy(ys)(0)
    return nn_ops.Ops.G


# Creates the graph for the model
def CreateGraph(graph_type, batch_size, hidden_dim_size, n_procs):
    nn_ops.Ops.default_procs = n_procs

    if graph_type == 'alexnet':
        G = AlexNet(batch_size)
    elif graph_type == 'resnet101':
        G = ResNet101(batch_size)
    elif graph_type == 'inception3':
        G = Inception3(batch_size)
    #elif graph_type == 'seq2seq':
    #    G = Seq2seq(batch_size)
    elif graph_type == 'transformer':
        G = Transformer(batch_size)
    elif graph_type == 'rnnlm':
        G = RNNLM(batch_size)
    else:
        assert False

    return G

