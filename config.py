import math
import itertools
from functools import reduce
import operator as op

def GetNextPartition(lst):
    if len(lst) == 1:
        yield [ lst ]
        return

    first = lst[0]
    for smaller in GetNextPartition(lst[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset
        yield [ [ first ] ] + smaller


def GetConfigs(node_dim_lst, n_procs):
    total_dims = reduce(operator.add, node_dim_lst, 0);
    partial_sums = [0] + list(itertools.accumulate(node_dim_lst))

    log_n_procs = int(math.log2(n_procs))
    procs = [1 << i for i in range(log_n_procs+1)];

    configs = []
    for flat_config in itertools.product(procs, repeat=total_dims):
        config = [list(flat_config[partial_sums[i]:partial_sums[i+1]]) for i in
                range(len(partial_sums) - 1)]
        used_procs = [reduce(operator.mul, c, 1) for c in config]
        used_procs = reduce(operator.add, used_procs, 0)

        if used_procs <= n_procs:
            configs.append(config)

    return configs


def GetNodeConfigs(node_dom, n_procs):
    dim = len(node_dom)
    log_n_procs = int(math.log2(n_procs))
    procs = [1 << i for i in range(log_n_procs + 1)]

    configs = []
    for c in itertools.product(procs, repeat=dim):
        used_procs = reduce(op.mul, c, 1)
        if used_procs <= n_procs:
            configs.append(c)

    return configs


#def main():
#    n_procs = 2;
#
#    for part in GetNextPartition([1, 2, 3]):
#        part_cfgs = []
#        for s in part:
#            print(str(s) + ":")
#            if len(s) > n_procs:
#                break;
#            cfgs = GetCfgs(s, n_procs);
#            [print(cfg) for cfg in cfgs]
#            print("\n")
#            part_cfgs.append(cfgs)
#
