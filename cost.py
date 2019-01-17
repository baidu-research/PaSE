import math
import numpy as np
from functools import reduce
import operator


def RoundUp(a, b):
    return np.ceil(a / b);


def ListDiv(a, b):
    return [x/y for x, y in zip(a, b)];


def ArrayDiv(a, b):
    return a/b


def CompCost(layer_dom, config_dom):
    assert(len(layer_dom) == len(config_dom));

    # Fw pass involves 1 matmult, and bw pass involves 2
    #cost = 3.0 * np.multiply.reduce(ArrayDiv(layer_dom, config_dom));
    cost = 3.0 * reduce(operator.mul, ListDiv(layer_dom, config_dom), 1);
    return cost;


def GetAreaNeeded(reqd, intersection):
    area_reqd = reduce(operator.mul, reqd, 1);
    area_intersection = 1.0;
    for x, y in zip(reqd, intersection):
        area_intersection *= min(x, y);

    assert(area_reqd >= area_intersection);
    return (area_reqd - area_intersection);


def FwdCommCost(layer1, layer2, config1, config2):
    cost = float(0);

    # Cost of replicating the inputs when no. of procs for layer1 is less than
    # no. of procs for layer2
    area = RoundUp(layer2.m, config2.m) * RoundUp(layer2.k, config2.k)
    cost += area

    # Cost of distributing inputs
    area_needed = GetAreaNeeded([RoundUp(layer2.m, config2.m), RoundUp(layer2.k,
        config2.k)], [RoundUp(layer1.m, config1.m), RoundUp(layer1.k, config1.k)])
    if area_needed > 0:
        cost += area_needed;

    # Cost for reducing the output
    if config2.k > 1:
        # All-reduce cost = 2*(N/P)*(P-1)
        words = RoundUp((RoundUp(layer2.m, config2.m) * RoundUp(layer2.n,
            config2.n)), config2.k);
        steps = 2.0 * (config2.k - 1);
        cost += (words * steps)

    return cost;


def BwdCommCost(layer1, layer2, config1, config2):
    cost = float(0);

    # Cost for propagating input gradients to previous layer
    area_needed = GetAreaNeeded([RoundUp(layer2.m, config2.m), RoundUp(layer2.k,
        config2.k)], [RoundUp(layer1.m, config1.m), RoundUp(layer1.n, config1.n)])
    if area_needed > 0:
        cost += area_needed

    # Cost of updating gradients
    if config2.m > 1:
        # All-reduce cost = 2*(N/P)*(P-1)
        words = RoundUp((RoundUp(layer2.k, config2.k) * RoundUp(layer2.n,
            config2.n)), config2.m);
        steps = 2.0 * (config2.m - 1);
        cost += (words * steps)

    return cost;


def CommCost(layer1, layer2, config1, config2):
    return (FwdCommCost(layer1, layer2, config1, config2) + BwdCommCost(layer1,
        layer2, config1, config2));


