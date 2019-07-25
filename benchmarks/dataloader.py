import os
import numpy as np
import tensorflow as tf


class ImageDataLoader():
    def parse_image(self, filename, label):
        image_string = tf.read_file(filename)
    
        image = tf.image.decode_jpeg(image_string, channels=3)
    
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
    
        image = tf.image.resize_images(image, [self.img_size[0],
            self.img_size[1]])
        label = tf.strings.to_number(label, out_type=tf.int32)
    
        return image, label

    def __init__(self, batch_size, img_size, dataset_size=1000,
            dataset_dir=None, labels_filename=None, num_parallel_calls = 32,
            prefetches = 8):
        assert len(img_size) == 2
        self.img_size = img_size

        if dataset_dir is None:
            self.dataset_size = dataset_size
            num_classes = 1000

            num_elems = 1000
            assert dataset_size % num_elems == 0
            features = tf.random_uniform([num_elems, img_size[0],
                img_size[1], 3], minval=0, maxval=1, dtype=tf.float32)
            classes = tf.random_uniform([num_elems], minval=0,
                    maxval=num_classes, dtype=tf.int32)
            dataset = tf.data.Dataset.from_tensor_slices((features,
                classes)).repeat(int(dataset_size / num_elems))

        else:
            assert dataset_dir is not None
            assert labels_filename is not None
            labels_filename = os.path.join(dataset_dir, labels_filename)

            filenames = []
            labels = []
            with open(labels_filename, 'r') as label_file:
              for line in label_file:
                s = line.split(' ')
                filenames.append(dataset_dir + '/train/' + s[0])
                labels.append(s[1])

            self.dataset_size = len(labels)

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
    def parse_text(self, sentence, label=None):
        # Split sentence into words, and convert it into ids
        sentence_split = tf.string_split([sentence]).values
        if self.max_seq_len is not None: # Trim the sentence to max_seq_len
            sentence_split = sentence_split[:self.max_seq_len]
        src_seq_len = tf.size(sentence_split)
        sentence = self.src_vocab.lookup(sentence_split)

        if label is not None:
            label_split = tf.string_split([label]).values
        else:
            label_split = sentence_split[1:]
        if self.max_seq_len is not None:
            label_split = label_split[:self.max_seq_len]
        tgt_seq_len = tf.size(label_split)
        label = self.tgt_vocab.lookup(label_split)

        # Prepend and append SOS and EOS tokens to label
        #label = tf.concat([[self.tgt_sos_token], label, [self.tgt_eos_token]],
        #        0)

        return sentence, label, src_seq_len, tgt_seq_len

    def __init__(self, batch_size, src_vocab_filename, tgt_vocab_filename,
            src_text_filename, tgt_text_filename, max_seq_len = None,
            num_parallel_calls = 32, prefetches = 8):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # Vocab to id table
        src_vocab = tf.contrib.lookup.index_table_from_file(src_vocab_filename)
        tgt_vocab = tf.contrib.lookup.index_table_from_file(
                tgt_vocab_filename) if tgt_vocab_filename is not None \
                        else src_vocab
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        def Lookup(vocab, word, default=tf.constant(0, tf.int64)):
            try:
                return vocab.lookup(tf.constant(word))
            except ValueError:
                return default

        self.src_pad_id = Lookup(src_vocab, '<pad>')
        self.tgt_pad_id = Lookup(tgt_vocab, '<pad>')
        #self.src_sos = Lookup(src_vocab, '<sos>')
        #self.tgt_sos = Lookup(tgt_vocab, '<sos>')
        #self.src_eos = Lookup(src_vocab, '<eos>')
        #self.tgt_eos = Lookup(tgt_vocab, '<eos>')

        # Sentences and labels datasets
        num_parallel_reads = 8
        sentences = tf.data.TextLineDataset(src_text_filename)
        if tgt_text_filename is not None:
            labels = tf.data.TextLineDataset(tgt_text_filename)
            dataset = tf.data.Dataset.zip((sentences, labels))
        else:
            dataset = tf.data.Dataset.zip(sentences)

        dataset = dataset.map(self.parse_text, num_parallel_calls =
                num_parallel_calls)

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
        self.tbl_initialized = False

    def next_batch(self):
        return self.dataset_iterator.get_next()

    def reset_pointer(self):
        sess = tf.get_default_session()
        if not self.tbl_initialized:
            sess.run(tf.tables_initializer())
            self.tbl_initialized = True
        sess.run(self.initializer)

