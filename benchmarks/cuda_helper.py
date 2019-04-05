import os
import torch
import torch.nn as nn


class CudaHelper():
    def __init__(self, n_procs):
        assert(torch.cuda.is_available())
        assert(n_procs <= torch.cuda.device_count())

        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in
            range(n_procs)])

        self.n_procs = n_procs
        self.device_ids = list(range(n_procs))
        self.devices = [torch.device("cuda:" + str(i)) for i in self.device_ids]

        self.default_device_id = torch.cuda.current_device()
        self.default_device = torch.device("cuda:" +
                str(self.default_device_id))


def ModelParallelLinear(dim1, dim2, devices, pointwise_ops=None):
    models = []
    for d in devices:
        model = nn.Linear(dim1, dim2).cuda(d)
        if pointwise_ops is not None:
            model = nn.Sequential(model, *pointwise_ops)
        models.append(model)

    return models