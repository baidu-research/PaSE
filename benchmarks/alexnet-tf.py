import sys
import time
from argparse import ArgumentParser
import functools

import numpy as np
import tensorflow as tf


def get_gpu_device(idx):
    return tf.device(tf.DeviceSpec(device_type = "GPU", device_item = idx))

class AlexNet(object):
  
  def __init__(self, x, keep_prob, num_classes, num_gpus):
    
    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.NUM_GPUS = num_gpus
    
    # Call the create function to build the computational graph of AlexNet
    self.create()
    
  def create(self):
    X = tf.split(self.X, self.NUM_GPUS)
    
    flattened_split = []
    for i in range(self.NUM_GPUS):
        with get_gpu_device(i):
            with tf.variable_scope(tf.get_variable_scope(), reuse =
                    tf.AUTO_REUSE):
                # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
                conv1 = conv(X[i], 11, 11, 96, 4, 4, padding = 'SAME', name = 'conv1')
                pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
                norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')
                
                # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
                conv2 = conv(norm1, 5, 5, 192, 1, 1, padding = 'SAME', name = 'conv2')
                pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
                norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
                
                # 3rd Layer: Conv (w ReLu)
                conv3 = conv(norm2, 3, 3, 384, 1, 1, padding = 'SAME', name = 'conv3')
                
                # 4th Layer: Conv (w ReLu) splitted into two groups
                conv4 = conv(conv3, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv4')
                
                # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
                conv5 = conv(conv4, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv5')
                pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

                pool6 = avg_pool(pool5, 8, 8, 1, 1, padding = 'VALID', name = 'pool6')

                flattened = tf.reshape(pool6, [-1, 6*6*256], name = 'flatten')
                flattened = tf.split(flattened, 4, axis = 1, name =
                        'flatten_split')

                flattened_split.append(flattened)

    flattened = []
    for i in range(4):
        with get_gpu_device(i):
            with tf.variable_scope(tf.get_variable_scope(), reuse =
                    tf.AUTO_REUSE):
                flattened = tf.concat(flattened_split[i], axis = 0, name =
                        'concat' + str(i))

    fn_x = lambda T, i, j: return T[j]

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    assert(len(flattened) == 4)
    fn_x1 = functools.partial(fn_x, flattened)
    gpu_id = lambda i, j: i * 4 + j
    fc6_split = model_par_fc(fn_x1, 6 * 6 * 256, 4096, 4, 2, gpu_id, name = 'fc6',
            keep_prob = self.KEEP_PROB)
    
    # 7th Layer: FC (w ReLu) -> Dropout
    assert(len(fc6_split) == 2)
    fn_x2 = functools.partial(fn_x, fc6_split)
    gpu_id = lambda i, j: i + j * 4
    fc7_split = model_par_fc(fn_x2, 4096, 4096, 2, 4, gpu_id, name = 'fc7',
            keep_prob = self.KEEP_PROB)
    
    # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
    assert(len(fc7_split) == 4)
    fn_x3 = functools.partial(fn_x, fc7_split)
    gpu_id = lambda i, j: i * 4 + j
    fc8_split = model_par_fc(fn_x3, 4096, self.NUM_CLASSES, 4, 2, gpu_id, name =
            'fc8', dropout = False, relu = False)

    assert(len(fc8_split) == 2)
    self.fc8 = tf.concat(fc8_split, axis = 1)
    
     
  
"""
Predefine all necessary layer for the AlexNet
""" 
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME'):
  """
  Adapted from: https://github.com/ethereon/caffe-tensorflow
  """
  # Get number of input channels
  input_channels = int(x.get_shape()[-1])
  
  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)
  
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])  
    
    conv = convolve(x, weights)
      
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)
        
    return make_data_parallel(relu)

def model_par_fc(fn_x, num_in, num_out, in_ch_parts, out_ch_parts, gpu_id, name,
        keep_prob = None, dropout = True, relu = True):
    out_split = []
    for i in range(out_ch_parts):
        suffix = '_' + str(in_ch_idx)
        fc_split = []

        for j in range(in_ch_parts):
            suffix = '_' + str(in_ch_idx) + '_' + str(out_ch_idx)

            with get_gpu_device(gpu_id(i, j)):
                with tf.variable_scope(str(tf.get_variable_scope()) + suffix,
                        reuse=tf.AUTO_REUSE):
                    X = fn_x(i, j)
                    fc_split.append(fc(X, num_in / in_ch_parts, num_out /
                        out_ch_parts, name = name + suffix, relu))

        # Partial reduction along k-dim
        with get_gpu_device(gpu_id(i, 0)):
            with tf.variable_scope(tf.get_variable_scope(),
                    reuse=tf.AUTO_REUSE):
                fc_split = tf.stack(fc_split, name = name + "_stack" + suffix)
                out = tf.reduce_sum(fc_split, axis = 0, keepdims = False, name =
                        name + "_reduce_sum" + suffix)

                if dropout:
                    out = dropout(out, keep_prob, name = "_dropout" + suffix)

                out_split.append(out)

    return out_split
    

def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)      
      return relu
    else:
      return act
    

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)
  
def avg_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.avg_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)
  
def lrn(x, radius, alpha, beta, name, bias=1.0):
  return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)
  
def dropout(x, keep_prob, name):
  return tf.nn.dropout(x, keep_prob, name = name)


class ImageDataGenerator():
    def __init__(data_size, num_classes):
        self.data_size = data_size
        self.num_classes = num_classes

    def next_batch(batch_size):
        data = np.random.uniform(size=[batch_size, 227, 227, 3])
        labels = np.randint(0, num_classes, size=batch_size)
        return data, labels
  
    
def main():
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, required=False, default=256,
            help="Batch size. (Default: 256)")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors. (Default: 8)")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
            help="No. of epochs")
    parser.add_argument('-d', '--dropout', type=float, required=False,
            default=0.5, help="keep_prob value for dropout layers. (Default:
            0.5)")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(2)), 
            help="Strategy to use. 0: DataParallel, 1: Optimized. (Default: 0)")
    args = vars(parser.parse_args())

    # Input parameters
    num_gpus = args['procs']
    batch_size = args['batch_size']
    dropout_rate = args['dropout']
    num_epochs = args['epochs']
    num_classes = 1000
    
    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [batch_size, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    model = AlexNet(x, keep_prob, num_classes, num_gpus)
    
    # Link variable to model output
    score = model.fc8
    
    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  
    
    # Train op
    with tf.name_scope("train"):
      # Create optimizer and apply gradient descent to the trainable variables
      train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
              colocate_gradients_with_ops = True)
    
    if log_summary:
        # Add the loss to summary
        tf.summary.scalar('cross_entropy', loss)
    
    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
      correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
     
    if log_summary:
        # Add the accuracy to the summary
        tf.summary.scalar('accuracy', accuracy)
        
        # Merge all summaries together
        merged_summary = tf.summary.merge_all()
        
        # Initialize the FileWriter
        writer = tf.summary.FileWriter(filewriter_path)
    
    # Initalize the data generator seperately for the training and validation set
    train_generator = ImageDataGenerator(batch_size, num_classes)
    val_generator = ImageDataGenerator(batch_size, num_classes)
    
    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)
    
    # Start Tensorflow session
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
     
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      
      # Add the model graph to TensorBoard
      if log_summary:
          writer.add_graph(sess.graph)
      
      print("{} Start training...".format(datetime.now()))
      print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                        filewriter_path))
      
      # Loop over number of epochs
      for epoch in range(num_epochs):
        
            print("{} Epoch number: {}".format(datetime.now(), epoch+1))
            
            step = 1
            
            while step < train_batches_per_epoch:
                
                # Get a batch of images and labels
                batch_xs, batch_ys = train_generator.next_batch(batch_size)
                
                # And run the training op
                sess.run(train_op, feed_dict={x: batch_xs, 
                                              y: batch_ys, 
                                              keep_prob: dropout_rate})
                
                # Generate summary with the current batch of data and write to file
                if step%display_step == 0:
                    if log_summary:
                        s = sess.run(merged_summary, feed_dict={x: batch_xs, 
                                                                y: batch_ys, 
                                                                keep_prob: 1.})
                        writer.add_summary(s, epoch*train_batches_per_epoch + step)
                    
                step += 1
                
            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.now()))
            test_acc = 0.
            test_count = 0
            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty = val_generator.next_batch(batch_size)
                acc = sess.run(accuracy, feed_dict={x: batch_tx, 
                                                    y: batch_ty, 
                                                    keep_prob: 1.})
                test_acc += acc
                test_count += 1
            test_acc /= test_count
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
            
            # Reset the file pointer of the image data generator
            val_generator.reset_pointer()
            train_generator.reset_pointer()
            
            print("{} Saving checkpoint of model...".format(datetime.now()))  
            
            #save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)  
            
            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        


if __name__ == "__main__":
    main()
