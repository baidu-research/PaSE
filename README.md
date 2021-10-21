# PaSE: Parallelization Strategies for Efficient DNN Training
PaSE is a tool that automatically computes efficient parallelization strategies for training DNNs in a multi-node/multi-GPU environment. There are several choices to parallelize each layer in a DNN. Exhaustively searching this space to find an optimal parallelization strategy is time consuming and impractical. PaSE uses a dynamic programming based approach to find an efficient strategy within a reasonable time. Please refer to our [paper](https://github.com/baidu-research/PaSE/raw/master/docs/PaSE_ipdps2021.pdf) for more details. (Published version of the paper can also be found on the IEEE website [here](https://ieeexplore.ieee.org/document/9460527).)

BibTeX:
```
@INPROCEEDINGS{9460527,
  author={Elango, Venmugil},
  booktitle={2021 IEEE International Parallel and Distributed Processing Symposium (IPDPS)}, 
  title={Pase: Parallelization Strategies for Efficient DNN Training}, 
  year={2021},
  volume={},
  number={},
  pages={1025-1034},
  doi={10.1109/IPDPS49936.2021.00111}}
```

## Getting started
Packages required to run the tool can be installed as follows:
```sh
> python3 -m venv ~/env/pase
> source ~/env/pase/bin/activate
> pip install -r requirements.txt
```
PaSE expects a computation graph of a DNN as input. Graphs for some of the common DDNs are defined in graph.py.
Usage of the tool is as follows:
```sh
> python3 ./scheduler.py --help
usage: scheduler.py [-h] [-p PROCS] [-b BATCH] [-m MODEL] [-g {alexnet,resnet101,inception3,rnnlm,transformer}] [-a {0,1}] [--profile] [--measure] [-d] [--algo {0,1}]
```
```
optional arguments:
  -h, --help            show this help message and exit
  -p PROCS, --procs PROCS
                        No. of processors. (Default: 32)
  -b BATCH, --batch BATCH
                        Batch size. (Default: 128)
  -m MODEL, --model MODEL
                        Model size. (Default: 128)
  -g {alexnet,resnet101,inception3,rnnlm,transformer}, --graph {alexnet,resnet101,inception3,rnnlm,transformer}
                        Neural net graph. (Default: 'alexnet')
  --flops FLOPS         Peak FLOPS of each device in TFLOPS. (default: 10.0)
  --bw BW               Peak inter-connection bandwidth in GBytes/sec (default: 16.0)
  --profile             Turn on/off profiling.
  --measure             Turn on/off measurement.
  -d, --dump-graph      Dump the graph in dot format to the file graph.dot in the working directory.
  --algo {0,1}          Algorithm to be used to compute strategy (Default: 0).
```

## Benchmarks
The _benchmarks_ folder contains parallel implementations of some of the common DNN models parallelized using the strategies suggested by PaSE, baseline data parallelism, and using expert defined strategies. These benchmarks are written in Mesh-Tensorflow (https://github.com/tensorflow/mesh). To install the packages required to run these benchmarks, run
```sh
> cd benchmarks
> pip install -r requirements.txt
```
