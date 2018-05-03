#NOTE: refer to this url: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
import tensorflow as tf
import os
import matplotlib.pyplot as plt


HEIGHT = 256
WIDTH = 256

def tfrec_data_input_fn(filenames, num_epochs=1, batch_size=16):
    
    def _input_fn():
        def _parse_record(tf_record):
            features = {
                'image': tf.FixedLenFeature([], dtype=tf.string),
                'label': tf.FixedLenFeature([], dtype=tf.int64)
            }
            record = tf.parse_single_example(tf_record, features)

            image_raw = tf.decode_raw(record['image'], tf.uint8)
            image_raw = tf.reshape(image_raw, shape=(256, 256))

            label = tf.cast(record['label'], tf.int32)
            

            return { 'image': image_raw }, label
        
        # For TF dataset blog post, see https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_record)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels
    
    return _input_fn

tfrec_dev_input_fn = tfrec_data_input_fn(["tftrain.tfrecords"])
features, labels = tfrec_dev_input_fn()

with tf.Session() as sess:
    for j in range(44):
        img, label = sess.run([features['image'], labels])
    '''for i in range(16):
        plt.imshow(img[i])
        plt.show()'''
        