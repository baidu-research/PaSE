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

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
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
            help="Strategy to be used. 0: OWT, 1: Optimized. (Default: 0)")
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
    model = AlexNet(num_classes)
    model.cuda()
    model = nn.DataParallel(model, device_ids=cu_helper.device_ids)
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
