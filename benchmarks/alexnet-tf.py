import sys
import time
import os
from datetime import datetime
from argparse import ArgumentParser
import functools

import numpy as np
import tensorflow as tf

def get_gpu_device(idx):
    return tf.device(tf.DeviceSpec(device_type = "GPU", device_index = idx))


def parse_image(filename, label):
  image_string = tf.read_file(filename)

  image = tf.image.decode_jpeg(image_string, channels=3)

  # This will convert to float values in [0, 1]
  image = tf.image.convert_image_dtype(image, tf.float32)

  image = tf.image.resize_images(image, [227, 227])
  label = tf.strings.to_number(label, out_type=tf.int32)

  return image, label


class AlexNet(object):
  
  def __init__(self, x, keep_prob, num_classes, num_gpus, strategy):
    
    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.NUM_GPUS = num_gpus
    
    # Call the create function to build the computational graph of AlexNet
    if strategy == 0:
        self.create_data_parallel()
    elif strategy == 1:
        assert (num_gpus == 4 or num_gpus == 8)
        if num_gpus == 4:
          self.create_optimized_4()
        else:
          self.create_optimized_8()
    else:
        assert(False)


  def create_data_parallel(self):
    X = tf.split(self.X, self.NUM_GPUS)
  
    out_split = []
    with tf.variable_scope('alexnet', reuse = tf.AUTO_REUSE):
        for i in range(self.NUM_GPUS):
            with get_gpu_device(i):
                # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
                conv1 = conv(X[i], 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
                pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
                norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')
                
                # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
                conv2 = conv(norm1, 5, 5, 256, 1, 1, name = 'conv2')
                pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
                norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
                
                # 3rd Layer: Conv (w ReLu)
                conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3')
                
                # 4th Layer: Conv (w ReLu) splitted into two groups
                conv4 = conv(conv3, 3, 3, 384, 1, 1, name = 'conv4')
                
                # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
                conv5 = conv(conv4, 3, 3, 256, 1, 1, name = 'conv5')
                pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')
                
                # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
                flattened = tf.reshape(pool5, [-1, 6*6*256])
                fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
                dropout6 = dropout(fc6, self.KEEP_PROB, name = 'dropout6')
                
                # 7th Layer: FC (w ReLu) -> Dropout
                fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
                dropout7 = dropout(fc7, self.KEEP_PROB, name = 'dropout7')
                
                # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
                fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')
                out_split.append(fc8)

        self.fc8 = tf.concat(out_split, axis = 0)

    
  def create_optimized_4(self):
    X = tf.split(self.X, self.NUM_GPUS)
    
    flattened_split = []
    with tf.variable_scope('alexnet', reuse = tf.AUTO_REUSE):
        for i in range(self.NUM_GPUS):
            with get_gpu_device(i):
                # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
                conv1 = conv(X[i], 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
                pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
                norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')
                
                # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
                conv2 = conv(norm1, 5, 5, 256, 1, 1, padding = 'SAME', name = 'conv2')
                pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
                norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
                
                # 3rd Layer: Conv (w ReLu)
                conv3 = conv(norm2, 3, 3, 384, 1, 1, padding = 'SAME', name = 'conv3')
                
                # 4th Layer: Conv (w ReLu) splitted into two groups
                conv4 = conv(conv3, 3, 3, 384, 1, 1, padding = 'SAME', name = 'conv4')
                
                # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
                conv5 = conv(conv4, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv5')
                pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

                flattened = tf.reshape(pool5, [-1, 6*6*256], name = 'flatten')
                flattened = tf.split(flattened, 4, axis = 1, name =
                        'flatten_split')

                flattened_split.append(flattened)

        flattened = []
        for i in range(4):
            with get_gpu_device(i):
                with tf.variable_scope('concat_' + str(i), reuse =
                        tf.AUTO_REUSE):
                    lst = [x[i] for x in flattened_split]
                    flattened.append(tf.concat(lst, axis = 0))

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        assert(len(flattened) == 4)
        gpu_id = lambda i, j: i * 4 + j
        fc6_split = model_par_fc(flattened, 6 * 6 * 256, 4096, 4, 1, gpu_id, name =
                'fc6', keep_prob = self.KEEP_PROB)
        
        # 7th Layer: FC (w ReLu) -> Dropout
        assert(len(fc6_split) == 1)
        gpu_id = lambda i, j: i + j * 4
        fc7_split = model_par_fc(fc6_split, 4096, 4096, 1, 4, gpu_id, name = 'fc7',
                keep_prob = self.KEEP_PROB)
        
        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        assert(len(fc7_split) == 4)
        gpu_id = lambda i, j: i * 4 + j
        fc8_split = model_par_fc(fc7_split, 4096, self.NUM_CLASSES, 4, 1, gpu_id,
                name = 'fc8', has_dropout = False, relu = False)

        assert(len(fc8_split) == 1)
        self.fc8 = tf.concat(fc8_split, axis = 1)

    
  def create_optimized_8(self):
    X = tf.split(self.X, self.NUM_GPUS)
    
    flattened_split = []
    with tf.variable_scope('alexnet', reuse = tf.AUTO_REUSE):
        for i in range(self.NUM_GPUS):
            with get_gpu_device(i):
                # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
                conv1 = conv(X[i], 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
                pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
                norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')
                
                # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
                conv2 = conv(norm1, 5, 5, 256, 1, 1, padding = 'SAME', name = 'conv2')
                pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
                norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
                
                # 3rd Layer: Conv (w ReLu)
                conv3 = conv(norm2, 3, 3, 384, 1, 1, padding = 'SAME', name = 'conv3')
                
                # 4th Layer: Conv (w ReLu) splitted into two groups
                conv4 = conv(conv3, 3, 3, 384, 1, 1, padding = 'SAME', name = 'conv4')
                
                # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
                conv5 = conv(conv4, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv5')
                pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

                flattened = tf.reshape(pool5, [-1, 6*6*256], name = 'flatten')
                flattened = tf.split(flattened, 4, axis = 1, name =
                        'flatten_split')

                flattened_split.append(flattened)

        flattened = []
        for i in range(4):
            with get_gpu_device(i):
                with tf.variable_scope('concat_' + str(i), reuse =
                        tf.AUTO_REUSE):
                    lst = [x[i] for x in flattened_split]
                    flattened.append(tf.concat(lst, axis = 0))

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        assert(len(flattened) == 4)
        gpu_id = lambda i, j: i * 4 + j
        fc6_split = model_par_fc(flattened, 6 * 6 * 256, 4096, 4, 2, gpu_id, name =
                'fc6', keep_prob = self.KEEP_PROB)
        
        # 7th Layer: FC (w ReLu) -> Dropout
        assert(len(fc6_split) == 2)
        gpu_id = lambda i, j: i + j * 4
        fc7_split = model_par_fc(fc6_split, 4096, 4096, 2, 4, gpu_id, name = 'fc7',
                keep_prob = self.KEEP_PROB)
        
        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        assert(len(fc7_split) == 4)
        gpu_id = lambda i, j: i * 4 + j
        fc8_split = model_par_fc(fc7_split, 4096, self.NUM_CLASSES, 4, 2, gpu_id,
                name = 'fc8', has_dropout = False, relu = False)

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
  
  with tf.variable_scope(name, reuse = tf.AUTO_REUSE) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])  
    
    conv = tf.nn.conv2d(x, weights, strides = [1, stride_y, stride_x, 1],
            padding = padding)
      
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name + "_relu")
        
    return relu

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
  
def lrn(x, radius, alpha, beta, name, bias=1.0):
  return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)
  
def dropout(x, keep_prob, name):
  return tf.nn.dropout(x, keep_prob, name = name)


def model_par_fc(X, num_in, num_out, in_ch_parts, out_ch_parts, gpu_id, name,
        keep_prob = None, has_dropout = True, relu = True):
    out_split = []
    for i in range(out_ch_parts):
        suffix = '_' + str(i)
        fc_split = []

        for j in range(in_ch_parts):
            suffix = '_' + str(i) + '_' + str(j)

            with get_gpu_device(gpu_id(i, j)):
                with tf.variable_scope(name + suffix, reuse=tf.AUTO_REUSE) as scope:
                    fc_split.append(fc(X[j], num_in / in_ch_parts, num_out /
                        out_ch_parts, name = scope.name, relu = relu))

        # Partial reduction along k-dim
        with get_gpu_device(gpu_id(i, 0)):
            with tf.variable_scope(name + suffix, reuse=tf.AUTO_REUSE):
                fc_split = tf.stack(fc_split)
                out = tf.reduce_sum(fc_split, axis = 0, keepdims = False)

                if has_dropout:
                    out = dropout(out, keep_prob, name + "_dropout")

                out_split.append(out)

    return out_split
    

class ImageDataGenerator():
    #def __init__(self, data_size, num_classes):
    #    self.data_size = data_size
    #    self.num_classes = num_classes

    #    #self.data = np.random.uniform(size=[data_size, 227, 227, 3])
    #    #self.labels = np.random.randint(0, high=self.num_classes,
    #    #        size=[data_size, self.num_classes])

    #    self.data = np.random.normal(size=[data_size, 227, 227, 3], loc=127,
    #            scale=60)
    #    self.labels = np.random.randint(size=[data_size, self.num_classes], low
    #            = 0, high = num_classes - 1)

    #def next_batch(self, batch_size):
    #    return self.data, self.labels

    #def reset_pointer(self):
    #    pass

    #def __init__(self, dataset_dir, labels_filename):
    #    dataset = imagenet.get_split('train', dataset_dir, labels_filename,
    #            file_pattern='%s')
    #    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    #    [self.image, self.label] = provider.get(['image', 'label'])
    #    #assert(self.image.shape[0] == self.label.shape[0])

    #    self.data_size = 1000 #self.image.shape[0]
    #    self.idx = 0

    #def next_batch(self, batch_size):
    #    start = self.idx
    #    end = min(self.idx+batch_size, self.data_size)
    #    image = self.image[start:end]
    #    label = self.label[start:end]
    #    self.idx += 1
    #    return image, label

    #def reset_pointer(self):
    #    self.idx = 0

    def __init__(self, batch_size, dataset_dir, labels_filename):
      filenames = []
      labels = []
      with open(labels_filename, 'r') as label_file:
        for line in label_file:
          s = line.split(' ')
          filenames.append(dataset_dir + '/train/' + s[0])
          labels.append(s[1])

      self.data_size = len(labels)

      dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
      #dataset = dataset.shuffle(len(filenames))
      dataset = dataset.map(parse_image, num_parallel_calls=32)
      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(8)

      self.dataset_iterator = dataset.make_initializable_iterator()
      self.initializer = self.dataset_iterator.initializer

    def next_batch(self):
      return self.dataset_iterator.get_next()

    def reset_pointer(self):
      tf.get_default_session().run(self.initializer)
        
    
def main():
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, required=False, default=256,
            help="Batch size. (Default: 256)")
    parser.add_argument('-p', '--procs', type=int, required=False, default=8,
            help="No. of processors. (Default: 8)")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=100,
            help="No. of epochs")
    parser.add_argument('-d', '--dropout', type=float, required=False,
            default=0.5, help="keep_prob value for dropout layers. (Default: 0.5)")
    parser.add_argument('-s', '--strategy', type=int, required=False, default=0,
            choices=list(range(2)), 
            help="Strategy to use. 0: DataParallel, 1: Optimized. (Default: 0)")
    args = vars(parser.parse_args())

    # Input parameters
    num_gpus = args['procs']
    batch_size = args['batch']
    dropout_rate = args['dropout']
    num_epochs = args['epochs']
    strategy = args['strategy']
    num_classes = 1000
    learning_rate = 0.01
    log_summary = False
    display_step = 10
    warmup = 10

    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(i) + ',' for i in
                                                 range(num_gpus))[:-1]
    
    # Initalize the data generator seperately for the training and validation set
    #train_generator = ImageDataGenerator(batch_size, num_classes)
    #val_generator = ImageDataGenerator(batch_size, num_classes)
    dataset_dir = "/work/data/image/imagenet/imagenet_1_per_class/"
    labels_filename = dataset_dir + "train_1_per_class.txt"
    train_generator = ImageDataGenerator(batch_size, dataset_dir, labels_filename)
    
    # TF placeholder for graph input and output
    #x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    #y = tf.placeholder(tf.float32, [batch_size, num_classes])
    batch = train_generator.next_batch()
    x = batch[0]
    y = batch[1]
    x.set_shape([batch_size, 227, 227, 3])
    y.set_shape([batch_size])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    model = AlexNet(x, keep_prob, num_classes, num_gpus, strategy)
    
    # Link variable to model output
    score = model.fc8
    
    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits =
        score, labels = y))  
    
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
    
    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
    #val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

    tot_time = float(0)
    cnt = 0
    
    # Start Tensorflow session
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
     
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      
      # Add the model graph to TensorBoard
      if log_summary:
          writer.add_graph(sess.graph)
      
      print("{} Start training...".format(datetime.now()))
      
      # Loop over number of epochs
      for epoch in range(num_epochs):
            # Reset the file pointer of the image data generator
            train_generator.reset_pointer()
            step = 0

            while step < train_batches_per_epoch:
                start = time.time()
                loss_val, _ = sess.run([loss, train_op], feed_dict={keep_prob:
                                                                    dropout_rate})
                end = time.time()

                cnt += 1
                tot_time += (end - start)
                
                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    print("Epoch: " + str(epoch) + "; Loss: " + str(loss_val))
                    if log_summary:
                        s = sess.run(merged_summary, feed_dict={x: batch_xs, 
                                                                y: batch_ys, 
                                                                keep_prob: 1.})
                        writer.add_summary(s, epoch*train_batches_per_epoch + step)
                    
                step += 1
             
            
    #avg_time = tot_time / float(cnt - warmup)
    #print("Avg. time: " + str(avg_time) + " s")
    img_per_sec = (train_generator.data_size * num_epochs) / tot_time
    print("Throughout: " + str(img_per_sec) + " images / sec")


if __name__ == "__main__":
    main()