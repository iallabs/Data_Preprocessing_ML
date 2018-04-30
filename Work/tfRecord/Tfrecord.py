from random import shuffle, seed

import glob

import numpy as np

import cv2

import tensorflow as tf

import sys

import matplotlib.pyplot as plt



UR_DATA_PATH = ""

shuffle_data = True



# This function returns the different paths and corresponding labels

def read_paths(ur_path):

    #Reading the path of each data(ex: image) and then extracting the labels from path

    data_path = glob.glob(ur_path)

    labels = [int(i.split('.')[0][-1]) for i in data_path]

    return data_path, labels


def shuffling_data(data_path, labels):

    c = list(zip(data_path, labels))

    shuffle(c)

    #NOTE: data and labeld are tuples. Care about immutability 

    data, labeled = zip(*c) #NOTE:ith element in data correpand to ith label elmt

    return data, labeled


def split_data(training_prop,test_prop, data, labels, dev_prop=0):

    ''' 

    - training_prop is required .numerical float < 1

    - dev_prop is optional. Numerical float < 0.3

    - test_prop is required. Numerical float < 0.3

    - data and labels are lists 

    '''

    #NOTE: data is a list of paths of each image

    training_data = data[0:int(training_prop*len(data))]

    training_labels = labels[0:int(training_prop*len(labels))]



    if dev_prop != 0:

        dev_data = data[int(training_prop*len(data)):int((1-dev_prop)*len(data))]

        dev_labels = labels[int(training_prop*len(labels)):int((1-dev_prop)*len(labels))]

        test_data = data[int((1-dev_prop)*len(data)):]

        test_labels = labels[int((1-dev_prop)*len(labels)):]

        return training_data,training_labels, dev_data, dev_labels, test_data, test_labels

    else:

        test_data = data[int((1-test_prop)*len(data)):]

        test_labels = labels[int((1-test_prop)*len(labels)):]

        return training_data, training_labels, test_data, test_labels



# Load image 

def load_image(data, s_width, s_height):

    #cv2 load data from data (value of one path) 
    img_load = cv2.imread(data)
    img_after = cv2.resize(img_load, dsize=(s_width, s_height))

    img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    img_after = img_after.astype(np.float32)
    img_after = img_after.tostring()
    return img_after

a = load_image("pulmonary_chest_xray/ChinaSet_AllFiles/CXR_png/CHNCXR_0001_0.png", 1000, 1000)


def _int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def tfrecord_file(data, labeled, x_filename, s_width, s_height):

    #NOTE:data is a tuple (list) of paths of each image

    #NOTE: data can come from the dev-set or test-set

    #NOTE: Same for labeled



    file_path = x_filename

    #Open a TFRecordWriter

    writer = tf.python_io.TFRecordWriter(file_path)

    for i in range(1):

        new_img = load_image(data[i], s_width, s_height)

        new_label = labeled[i]

        new_feature = {'train/label': _int64_feature(new_label),

                        'train/image': _bytes_feature(tf.compat.as_bytes(new_img))
        }


        #NOTE: PLease refer to this url for defenirtion: https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto

        example = tf.train.Example(features=tf.train.Features(feature=new_feature))

        writer.write(example.SerializeToString())

    writer.close()

    sys.stdout.flush()

    return 0


data_path, labels = read_paths("pulmonary_chest_xray/ChinaSet_AllFiles/CXR_png/*.png")

tfrecord_file(data_path, labels, "Tfexamp.tfrecords", 1000, 1000)

'''data_test_path, test_labels = read_paths("pulmonary_chest_xray/MontgomerySet/CXR_png/*.png")

tfrecord_file(data_test_path, test_labels, "TfTestSet.tfrecords",1000,1000)'''



def read_file(x_filename, x_capacity=662):

    feature = {'train/image': tf.FixedLenFeature([], tf.string),

                'train/label': tf.FixedLenFeature([], tf.int64)

    }

    queue = tf.train.string_input_producer([x_filename], num_epochs=1)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(queue)

    features = tf.parse_single_example(serialized_example, feature=feature)

    

    #x means training, dev or test

    x_image = tf.decode_raw(features['train/image'], tf.float32)

    x_label = tf.cast(features['train/label'], tf.int64)


    #NOTE: In this line, we can add a instruction to preprocess the images to their original shape

    return x_image, x_label