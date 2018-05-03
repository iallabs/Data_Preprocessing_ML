import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#NOTE: refer to this url: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

def tfrec_data_input_fn(filenames, num_epochs=1, batch_size=16, shuffle=False):
    
    def _input_fn():
        def _parse_record(tf_record):
            features = {
                'image': tf.FixedLenFeature([], dtype=tf.string),
                'label': tf.FixedLenFeature([], dtype=tf.int64)
            }
            record = tf.parse_single_example(tf_record, features)

            image_raw = tf.decode_raw(record['image'], tf.uint8)
            label = tf.one_hot(tf.cast(record['label'], tf.int32), depth=2)
            
            def _normalize(image_x):
                """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
                image = tf.cast(image_x, tf.float32) * (1. / 255)-0.5
                return image
            
            image_raw = _normalize(image_raw)
            '''image_raw = tf.reshape(image_raw, (32,32,3))'''#For vizualizing 
            return { 'image': image_raw}, label
        
        # For TF dataset blog post, see https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_record)
        
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        '''features, labels = iterator.get_next()'''

        return iterator
    
    return _input_fn


def read_file(x_filename, x_capacity=800):
    record_iterator = tf.python_io.tf_record_iterator(path=x_filename)

    example = tf.train.Example()
    for str_rec in record_iterator:
        example.ParseFromString(str_rec)
        height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])

        width = int(example.features.feature['width']
                                .int64_list
                                .value[0])
    

        img_string = (example.features.feature['image']
                                  .bytes_list
                                  .value[0])

        label = (example.features.feature['label'].int64_list.value[0])


    return 0


def tfrec_data_catvdog(filenames, num_epochs=1, batch_size=16):
    
    def _input_fn():
        def _parse_record(tf_record):
            features = {
                'image': tf.FixedLenFeature([], dtype=tf.string),
            }
            record = tf.parse_single_example(tf_record, features)

            image_raw = tf.decode_raw(record['image'], tf.uint8)
            
            def _normalize(image):
                """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
                image = tf.cast(image, tf.float32) * (1. / 255)-0.5
                return image

            image_n = _normalize(image_raw)
            return { 'image': image_n }
        
        # For TF dataset blog post, see https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_record)

        '''dataset = dataset.repeat(num_epochs)'''
        '''dataset = dataset.batch(batch_size)'''

        iterator = dataset.make_one_shot_iterator()
        features= iterator.get_next()

        return features
    
    return _input_fn


'''read_file("./tftest.tfrecords")

tfrec_dev_input_fn = tfrec_data_input_fn(["tftest.tfrecords"])
features, labels = tfrec_dev_input_fn()

with tf.Session() as sess:
    for step in range(2):
        img, label = sess.run([features['image'], labels])
        img = np.reshape(img, (16,64,64,3))
    
        for i in range(16):
            plt.imshow(img[i])
            plt.show()'''
        