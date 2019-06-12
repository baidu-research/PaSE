import os
import tensorflow as tf


class ImageDataLoader():
    def parse_image(self, filename, label):
        image_string = tf.read_file(filename)
    
        image = tf.image.decode_jpeg(image_string, channels=3)
    
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
    
        image = tf.image.resize_images(image, [227, 227])
        label = tf.strings.to_number(label, out_type=tf.int32)
    
        return image, label

    def __init__(self, batch_size, dataset_dir, labels_filename,
            num_parallel_calls = 32, prefetches = 8):
        with tf.device('/cpu:0'):
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
            dataset = dataset.map(self.parse_image, num_parallel_calls =
                    num_parallel_calls)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(prefetches)

            self.dataset_iterator = dataset.make_initializable_iterator()
            self.initializer = self.dataset_iterator.initializer

    def next_batch(self):
        return self.dataset_iterator.get_next()

    def reset_pointer(self):
        tf.get_default_session().run(self.initializer)
 

class TextDataLoader():
    def parse_text(self, sentence, label):
        # Split sentence into words, and convert it into ids
        sentence = tf.string_split([sentence]).values
        src_seq_len = tf.size(sentence)
        if self.max_seq_len is not None:
            assert src_seq_len <= self.max_seq_len
        sentence = self.src_vocab.lookup(sentence)

        label = tf.string_split([label]).values
        tgt_seq_len = tf.size(label)
        if self.max_seq_len is not None:
            assert tgt_seq_len <= self.max_seq_len
        label = self.tgt_vocab.lookup(label)

        # Prepend and append SOS and EOS tokens to label
        #label = tf.concat([[self.tgt_sos_token], label, [self.tgt_eos_token]],
        #        0)

        return sentence, label, src_seq_len, tgt_seq_len

    def __init__(self, batch_size, src_vocab_filename, tgt_vocab_filename,
            src_text_filename, tgt_text_filename, max_seq_len=None,
            num_parallel_calls = 32, prefetches = 8):
        with tf.device('/cpu:0'):
            self.batch_size = batch_size
            self.max_seq_len = max_seq_len

            # Vocab to id table
            src_vocab = tf.contrib.lookup.index_table_from_file(src_vocab_filename)
            tgt_vocab = tf.contrib.lookup.index_table_from_file(tgt_vocab_filename)

            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

            # Sentences and labels datasets
            sentences = tf.data.TextLineDataset(src_text_filename)
            labels = tf.data.TextLineDataset(tgt_text_filename)
            dataset = tf.data.Dataset.zip((sentences, labels))

            dataset = dataset.map(self.parse_text, num_parallel_calls =
                    num_parallel_calls)

            self.src_pad_id = src_vocab.lookup(tf.constant('<pad>'))
            self.tgt_pad_id = tgt_vocab.lookup(tf.constant('<pad>'))

            # Shape: sentence, label, src_seq_len, tgt_seq_len
            padded_shapes = (tf.TensorShape([max_seq_len]),
                    tf.TensorShape([max_seq_len]), tf.TensorShape([]),
                    tf.TensorShape([]))
            padding_values = (self.src_pad_id, self.tgt_pad_id, tf.constant(0,
                dtype=tf.int32), tf.constant(0, dtype=tf.int32))

            #dataset = dataset.shuffle(buffer_size=buffer_size) 
            dataset = dataset.padded_batch(batch_size, padded_shapes =
                    padded_shapes, padding_values = padding_values,
                    drop_remainder = True)
            dataset = dataset.prefetch(prefetches)

            self.dataset_iterator = dataset.make_initializable_iterator()
            self.initializer = self.dataset_iterator.initializer

    def next_batch(self):
        return self.dataset_iterator.get_next()

    def reset_pointer(self):
        sess = tf.get_default_session()
        sess.run(tf.tables_initializer())
        sess.run(self.initializer)

