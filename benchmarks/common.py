import argparse
import numpy as np
import tensorflow.compat.v1 as tf

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
        parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
                help="No. of epochs")
        parser.add_argument('--max_steps', type=int, required=False,
                default=-1, help='Maximum no. of steps to execute')
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
        parser.add_argument('--src_vocab_size', type=int, required=False,
                default=None)
        parser.add_argument('--tgt_vocab_size', type=int, required=False,
                default=None)
        parser.add_argument('--sentences_size', type=int, required=False,
                default=None)
        parser.add_argument('--src_vocab', type=str, help="Source vocab data file.")
        parser.add_argument('--tgt_vocab', type=str, help="Target vocab data file.")
        parser.add_argument('--src_text', type=str, help="Source text data file.")
        parser.add_argument('--tgt_text', type=str, help="Target text data file.")
        parser.add_argument('--seq_len', type=int, required=False, default=256,
                help='Maximum sequence length')
    
        args = parser.parse_args()
        gpus_per_node = args.gpus
        self.num_nodes = int(os.environ['SLURM_NNODES'])
        assert self.num_nodes > 0
        self.num_gpus = gpus_per_node * self.num_nodes
        self.args = args
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
    
            cluster = tf.train.ClusterSpec({"localhost":hostlist_w_port}).as_cluster_def()
            server = tf.distribute.Server(cluster, job_name="localhost",
                    task_index=task_index, protocol='grpc+verbs')
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
    
    def train(self, init_ops, loss_op, weight_updates, dataset, config=None,
            run_options=None):
        config = config or tf.ConfigProto(allow_soft_placement=False)
        assert all(w is not None for w in weight_updates)

        # Workaround to prevent MPI from crashing due to 'StopIteration'
        if self.args.max_steps < 1:
            try:
                self.args.max_steps = dataset.dataset_size // self.args.batch_size
            except KeyError:
                pass

        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        cnt = 0
        with tf.variable_scope('train'):
            with tf.Session(self.session_target, config=config) as sess:
                dataset.reset_pointer(sess)
                sess.run(tf.global_variables_initializer())
                sess.run(init_ops)
                print('Finished initialization.')
    
                tot_time = float(0)
                start = time.time()
                for epoch in range(self.args.epochs):
                    step = 0
    
                    while True:
                        try:
                            loss_val, *_ = sess.run([loss_op] + weight_updates,
                                    options=run_options)
                            step += 1
                            cnt += 1

                            if step % self.args.display_steps == 0:
                                print("Epoch: " + str(epoch) + "; Loss: " +
                                        str(loss_val))

                            if self.args.max_steps > 0 and (step ==
                                    self.args.max_steps):
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

    #def train_model(self, graph, mesh_to_impl, loss_op, dataset, config=None,
    #        run_options=None):
    #    # Optimize
    #    grad_updates = utils.Optimize(graph, loss_op)

    #    # Lower
    #    print('Beginning to lower mtf graph...', flush=True)
    #    lowering = mtf.Lowering(graph, mesh_to_impl)
    #    print('Finished lowering.', flush=True)

    #    # Init, loss and gradients
    #    init_ops = lowering.copy_masters_to_slices()
    #    tf_loss = lowering.export_to_tf_tensor(mtf_loss)
    #    tf_grad_updates = [lowering.lowered_operation(
    #        op) for op in grad_updates] + graph.rnn_grad_ws

    #    return train(self, init_ops, tf_loss, grad_updates, dataset, config,
    #            run_options))
