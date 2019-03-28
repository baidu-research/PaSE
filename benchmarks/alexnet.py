import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cuda_helper

class AlexNetDataParallel(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetDataParallel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetOptimal(nn.Module):
    def __init__(self, device_ids, devices, num_classes=1000):
        super(AlexNetOptimal, self).__init__()
        self.device_ids = device_ids
        self.devices = devices

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Dropout(),
        )

        dim1 = int((256 * 6 * 6) / 4)
        dim2 = int(4096 / 2)
        self.classifier1 = cuda_helper.ModelParallelLinear(dim1, dim2, devices,
                pointwise_ops=[nn.ReLU(inplace=True), nn.Dropout()])

        dim1 = int(4096 / 2)
        dim2 = int(4096 / 4)
        self.classifier2 = cuda_helper.ModelParallelLinear(dim1, dim2, devices,
                pointwise_ops=[nn.ReLU(inplace=True)])

        dim1 = int(4096 / 4)
        dim2 = int(num_classes / 2)
        self.classifier3 = cuda_helper.ModelParallelLinear(dim1, dim2, devices)

    def forward(self, x):
        from torch.nn.parallel._functions import ReduceAddCoalesced
        xs = nn.parallel.scatter(x, self.device_ids)

        # Apply features
        features = nn.parallel.replicate(self.features, self.device_ids)
        xs = nn.parallel.parallel_apply(features, xs)
        assert(len(xs) % 2 == 0)

        # Distribute the inputs
        inputs = []
        for i in range(0, len(xs), 2):
            t1, t2 = xs[i:i+2]
            proc1, proc2 = self.devices[i:i+2]

            t = torch.cat([t1, t2.to(proc1, non_blocking=True)])
            t = t.view(-1, 256*6*6)
            inputs.append(t)

            t = torch.cat([t1.to(proc2, non_blocking=True), t2])
            t = t.view(-1, 256*6*6)
            inputs.append(t)
        torch.cuda.synchronize()
        xs = inputs

        # Apply the first classifier
        assert(len(self.classifier1) == len(xs))
        xs = nn.parallel.parallel_apply(self.classifier1, xs)

        # Reduce the outputs
        r1 = ReduceAddCoalesced.apply(self.devices[0], 4, xs[0::2])
        r2 = ReduceAddCoalesced.apply(self.devices[1], 4, xs[1::2])
        xs = [r1, r2, r1.to(self.devices[2]), r2.to(self.devices[3]),
                r1.to(self.devices[4]), r2.to(self.devices[5]),
                r1.to(self.devices[6]), r2.to(self.devices[7])]

        # Apply the second classifier
        assert(len(self.classifier2) == len(xs))
        xs = nn.parallel.parallel_apply(self.classifier2, xs)

        # Reduce the outputs
        r = []
        for i in range(0, 8, 2):
            x = ReduceAddCoalesced.apply(self.devices[i], 2, xs[i:i+2])
            r.append(x)
            r.append(x.to(self.device[i+1]))
        xs = r

        # Apply the third classifier
        assert(len(self.classifier3) == len(xs))
        xs = nn.parallel.parallel_apply(self.classifier3, xs)

        # Reduce the outputs
        r1 = ReduceAddCoalesced.apply(self.devices[0], 4, xs[0::2])
        r2 = ReduceAddCoalesced.apply(self.devices[1], 4, xs[1::2])
        xs = [r1, r2]

        # Gather the final output into device 0
        x = nn.parallel.gather(xs, self.devices[0])

        return x


def run(model, criterion, optimizer, x, labels):
    y = model(x)
    loss = criterion(y, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def main():
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, required=False, default=256,
            help="Batch size. (Default: 256)")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors. (Default: 8)")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
            help="No. of epochs")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(2)), 
            help="Strategy to be used. 0: DataParallel, 1: Optimized. (Default: 0)")
    args = vars(parser.parse_args())

    # Parameter values
    batch_size = args['batch']
    n_procs = args['procs']
    epochs = args['epochs']
    strategy = args['strategy']
    warmup = 4
    num_classes = 1000

    assert(batch_size % n_procs == 0)

    cu_helper = cuda_helper.CudaHelper(n_procs)

    # Model
    if strategy == 0:
        model = AlexNetDataParallel(num_classes)
        model = nn.DataParallel(model, device_ids=cu_helper.device_ids)
    else:
        assert(n_procs == 8) # TODO: currently only handling this case
        model = AlexNetOptimal(cu_helper.device_ids, cu_helper.devices,
                num_classes)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Input, label tensors
    x = torch.cuda.FloatTensor(batch_size, 3, 224, 224).normal_()
    labels = torch.cuda.LongTensor(batch_size).random_(0, num_classes)

    # Warmup runs
    for i in range(warmup):
        loss = run(model, criterion, optimizer, x, labels)

    # Training
    tot_time = float(0)
    cnt = 0
    for i in range(warmup, epochs):
        running_loss = 0.0

        start = time.time()
        loss = run(model, criterion, optimizer, x, labels)
        end = time.time()

        tot_time += (end - start)
        cnt += 1
        running_loss += loss.item()

        print("Loss: " + str(running_loss))

    avg_time = tot_time / float(cnt)
    print("Avg. time: " + str(avg_time) + " s")


if __name__ == "__main__":
    main()
