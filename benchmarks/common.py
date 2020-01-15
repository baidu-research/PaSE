import argparse
import numpy as np
import tensorflow.compat.v1 as tf

import datetime
import sys, time, os

import utils

class Trainer():
    def __init__(self, parser=None):
        parser = parser or argparse.ArgumentParser(
                formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-b', '--batch_size', type=int, required=False, default=256,
                help="Batch size.")
        parser.add_argument('-g', '--gpus', type=int, required=False, default=8,
                help="No. of GPUs per node.")
        parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
                help="No. of epochs")
        parser.add_argument('--display_steps', type=int, required=False, default=10,
                help="No. of epochs")
        parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
                choices=list(range(4)),
                help="Strategy to use. 0: DataParallel, \
                        1: Optimized, \
                        2: Expert (OWT), \
                        3: FlexFlow")
        parser.add_argument('--xla', action='store_true', help='Use TF XLA')
        parser.add_argument('--dataset_dir', type=str, required=False, default=None,
                help='Dataset directory')
        parser.add_argument('--labels_filename', type=str, required=False,
                default='labels.txt', help='Labels filename')
        parser.add_argument('--dataset_size', type=int, required=False,
                default=1000, help='Labels filename')
    
        args = parser.parse_args()
        gpus_per_node = args.gpus
        self.num_nodes = int(os.environ['SLURM_NNODES'])
        assert self.num_nodes > 0
        self.num_gpus = gpus_per_node * self.num_nodes
        self.args = args
    
        if gpus_per_node != 8:
            raise NotImplementedError('Current implementation only handles 8 GPUs.')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                     range(gpus_per_node))[:-1]

        if self.args.xla:
            os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=' + os.environ['CUDA_HOME']
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

        tf.disable_eager_execution()
        self.setup_servers()
    

    def setup_servers(self):
        if self.num_nodes > 1:
            from hostlist import expand_hostlist
            
            task_index = int(os.environ['SLURM_PROCID'])
            hostlist = expand_hostlist(os.environ['SLURM_NODELIST'])
            hostlist_w_port = [("%s:2222" % host) for host in hostlist] 
    
            cluster = tf.train.ClusterSpec({"worker":hostlist_w_port}).as_cluster_def()
            server = tf.distribute.Server(cluster, job_name="worker",
                    task_index=task_index)
            session_target = server.target
    
            if task_index != 0:
                utils.join_tasks(task_index, hostlist)
                quit()
    
        else:
            task_index = 0
            hostlist = ['localhost']
            session_target = ''
    
        [print(f'{arg} : {val}') for arg, val in vars(self.args).items()]
        self.session_target = session_target
        self.task_index = task_index
        self.hostlist = hostlist
    
    def train(self, init_ops, loss_op, grad_ops, dataset, 
            train_batches_per_epoch=-1, config=None, run_options=None):
        config = config or tf.ConfigProto()

        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        cnt = 0
        with tf.variable_scope('train'):
            with tf.Session(self.session_target, config=config) as sess:
                dataset.reset_pointer(sess)
                sess.run(init_ops)
                print('Finished initialization.')
    
                tot_time = float(0)
                start = time.time()
                for epoch in range(self.args.epochs):
                    step = 0
    
                    while True:
                        try:
                            loss_val, *_ = sess.run([loss_op] + grad_ops,
                                    options=run_options)
                            step += 1
                            cnt += 1

                            if step % self.args.display_steps == 0:
                                print("Epoch: " + str(epoch) + "; Loss: " +
                                        str(loss_val))

                            if train_batches_per_epoch > 0 and (step ==
                                    train_batches_per_epoch):
                                break
                        except (tf.errors.OutOfRangeError, StopIteration):
                            break
    
                    dataset.reset_pointer(sess)
                end = time.time()
                tot_time += (end - start)
    
                samples_per_sec = float(self.args.batch_size * cnt) / tot_time
                print("Throughput: " + str(samples_per_sec) + " samples / sec",
                        flush=True)
    
        utils.join_tasks(self.task_index, self.hostlist)
