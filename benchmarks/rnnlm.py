import numpy as np
import tensorflow.compat.v1 as tf

import common
import utils
from dataloader import TextDataLoader

class Params():
    def __init__(self, batch_size, vocab_size, max_seq_len, num_nodes, devices):
        self.batch_size = batch_size
        self.num_units = 2048
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = 2
        self.num_nodes = num_nodes
        self.devices = devices

def main():
    trainer = common.Trainer()
    args = trainer.args
    devices = [f'/job:localhost/replica:0/task:{i}/device:GPU:{j}' for i in
            range(trainer.num_nodes) for j in range(args.gpus)]

    # Initialize dataset
    dataset = TextDataLoader(args.batch_size, args.src_vocab, None,
            args.src_text, None, args.seq_len, args.src_vocab_size,
            args.tgt_vocab_size, args.sentences_size)
    inputs, labels, _, _ = dataset.next_batch()

    vocab_size = utils.RoundUp(dataset.src_vocab_size, 8)
    print("Vocab size: %d" % vocab_size)
    params = Params(args.batch_size, vocab_size, args.seq_len,
            trainer.num_nodes, devices)

    # Model
    if args.strategy == 0:
        import rnnlm_data
        init_ops, loss, grads = rnnlm_data.model(params, inputs, labels)
    elif args.strategy == 1:
        import rnnlm_opt
        loss, grads = rnnlm_opt.model(params, inputs, labels)
    elif args.strategy == 2:
        import rnnlm_gnmt
        loss, grads = rnnlm_gnmt.model(params, inputs, labels)
    elif args.strategy == 3:
        import rnnlm_flexflow
        loss, grads = rnnlm_flexflow.model(params, inputs, labels)
    else:
        assert False

    # Train
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    config = tf.ConfigProto(allow_soft_placement=False)
    trainer.train(init_ops, loss, [grads], dataset, config=config,
            run_options=run_options)


if __name__ == '__main__':
    main()

