import os
import tensorflow as tf


def parse_image(filename, label):
  image_string = tf.read_file(filename)

  image = tf.image.decode_jpeg(image_string, channels=3)

  # This will convert to float values in [0, 1]
  image = tf.image.convert_image_dtype(image, tf.float32)

  image = tf.image.resize_images(image, [227, 227])
  label = tf.strings.to_number(label, out_type=tf.int32)

  return image, label


class ImageDataLoader():
    def __init__(self, batch_size, dataset_dir, labels_filename,
            num_parallel_calls = 32, prefetches = 8):
      labels_filename = os.path.join(dataset_dir, labels_filename)

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
      dataset = dataset.map(parse_image, num_parallel_calls=num_parallel_calls)
      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(prefetches)

      self.dataset_iterator = dataset.make_initializable_iterator()
      self.initializer = self.dataset_iterator.initializer

    def next_batch(self):
      return self.dataset_iterator.get_next()

    def reset_pointer(self):
      tf.get_default_session().run(self.initializer)
 

