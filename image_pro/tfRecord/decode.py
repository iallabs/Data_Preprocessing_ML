import os
import numpy as np
import tensorflow as tf
from image_pro.parameters import *
from image_pro.tfRecord.prepro import *
#NOTE: refer to this url: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

def tfrec_data_input_fn(filenames, num_epochs=1, batch_size=16, shuffle=False, is_training=False):
    
    def _input_fn():
        def _parse_record(tf_record):
            features = {
                'image': tf.FixedLenFeature([], dtype=tf.string),
                'label': tf.FixedLenFeature([], dtype=tf.int64)
            }
            record = tf.parse_single_example(tf_record, features)

            image_raw = tf.decode_raw(record['image'], tf.uint8)
            image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)
            print(image_raw)
            image_raw = tf.reshape(image_raw, [T_HEIGHT, T_WIDTH, T_Channnels])
            label = tf.one_hot(tf.cast(record['label'], tf.int32), depth=2)
            
            """def _normalize(image_x):
                Convert `image` from [0, 255] -> [-0.5, 0.5] floats.
                image = tf.cast(image_x, tf.float32) * (1. / 255) - 0.5
                return image
            
            image_raw = _normalize(image_raw)"""
           
            return image_raw, label
        
        # For TF dataset blog post, see https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_record)
        if is_training:
            dataset = dataset.map(preprocess_image_train)
        else:
            dataset = dataset.map(preprocess_image_eval)
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


def tfrec_data_catvdog(filenames):
    
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
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        return features
    
    return _input_fn