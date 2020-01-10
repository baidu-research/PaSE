import argparse
import numpy as np
import tensorflow as tf

import datetime
import sys, time, os

import utils

class Trainer():
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class =
                argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-b', '--batch_size', type=int, required=False, default=256,
                help="Batch size.")
        parser.add_argument('-g', '--gpus', type=int, required=False, default=8,
                help="No. of GPUs per node.")
        parser.add_argument('-n', '--nodes', type=int, required=False, default=1,
                help="No. of nodes.")
        parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
                help="No. of epochs")
        parser.add_argument('--display_steps', type=int, required=False, default=10,
                help="No. of epochs")
        parser.add_argument('-d', '--dropout', type=float, required=False,
                default=0.5, help="keep_prob value for dropout layers. (Default: 0.5)")
        parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
                choices=list(range(4)),
                help="Strategy to use. 0: DataParallel, \
                        1: Optimized, \
                        2: Expert (OWT), \
                        3: FlexFlow")
        parser.add_argument('--dataset_dir', type=str, required=False, default=None,
                help='Dataset directory')
        parser.add_argument('--labels_filename', type=str, required=False,
                default='labels.txt', help='Labels filename')
        parser.add_argument('--dataset_size', type=int, required=False,
                default=1000, help='Labels filename')
    
        args = parser.parse_args()
        gpus_per_node = args.gpus
        num_nodes = args.nodes
        num_gpus = gpus_per_node * num_nodes
        self.args = args
    
        if gpus_per_node != 8:
            raise NotImplementedError('Current implementation only handles 8 GPUs.')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                     range(gpus_per_node))[:-1]
    
        if num_nodes > 1:
            from hostlist import expand_hostlist
            
            n_tasks = int(os.environ['SLURM_NNODES'])
            assert n_tasks == num_nodes
    
            task_index = int(os.environ['SLURM_PROCID'])
            hostlist = expand_hostlist(os.environ['SLURM_NODELIST'])
            hostlist_w_port = [("%s:2222" % host) for host in hostlist] 
    
            cluster = tf.train.ClusterSpec({"worker":hostlist_w_port}).as_cluster_def()
            server = tf.train.Server(cluster, job_name="worker",
                    task_index=task_index)
            session_target = server.target
    
            if task_index != 0:
                utils.join_tasks(task_index, hostlist)
                quit()
    
        else:
            task_index = 0
            hostlist = ['localhost']
            session_target = ''
    
        [print(f'{arg} : {val}') for arg, val in vars(args).items()]
        self.session_target = session_target
        self.task_index = task_index
        self.hostlist = hostlist
    
    def train(self, init_ops, loss_op, grad_ops, dataset):
        train_batches_per_epoch = np.floor(dataset.dataset_size /
                self.args.batch_size).astype(np.int16)
        assert train_batches_per_epoch > 0
        config = tf.ConfigProto()

        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.variable_scope('train'):
            with tf.Session(self.session_target, config=config) as sess:
                dataset.reset_pointer()
                sess.run(init_ops)
                print('Finished initialization.')
    
                tot_time = float(0)
                start = time.time()
                for epoch in range(self.args.epochs):
                    step = 0
    
                    for _ in range(train_batches_per_epoch):
                        loss_val, *_ = sess.run([loss_op] + grad_ops)
                        step += 1
    
                        if step % self.args.display_steps == 0:
                            print("Epoch: " + str(epoch) + "; Loss: " +
                                    str(loss_val))
    
                    dataset.reset_pointer()
                end = time.time()
                tot_time += (end - start)
    
                img_per_sec = float(dataset.dataset_size * self.args.epochs) / tot_time
                print("Throughput: " + str(img_per_sec) + " images / sec",
                        flush=True)
    
        utils.join_tasks(self.task_index, self.hostlist)
