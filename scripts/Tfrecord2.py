from random import shuffle, seed

import glob

import numpy as np

import cv2

import tensorflow as tf

import sys

import matplotlib.pyplot as plt



UR_DATA_PATH = "pulmonary-chest-xray/*/CXR_png/*.png"

UR_TEST_SIZE = 100




# This function returns the different paths and corresponding labels
def shuffling_data(data_path, labels):
    seed(3)
    c = list(zip(data_path, labels))

    shuffle(c)

    #NOTE: data and labeld are tuples. Care about immutability 

    data, labeled = zip(*c) #NOTE:ith element in data correpand to ith label elmt

    return data, labeled, len(data), len(labeled)

def read_paths_treat(ur_path, test_size):

    #Reading the path of each data(ex: image) and then extracting the labels from path
    data_path = glob.glob(ur_path)
    labels = [int(i.split('.')[0][-1]) for i in data_path]
    data_path_shuffled, labels_shuffled, len_d, len_l = shuffling_data(data_path, labels)
    data_path_test, labels_test = data_path_shuffled[len_d-test_size: ], labels_shuffled[len_l-test_size: ]
    data_path_train,labels_train = data_path_shuffled[0:len_d-test_size ], labels_shuffled[0:len_l-test_size]
    return data_path_test, labels_test, data_path_train, labels_train






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

    img_after = cv2.cvtColor(img_load, cv2.COLOR_BGR2GRAY)
    
    img= cv2.resize(img_after, dsize=(s_width, s_height), interpolation=cv2.INTER_LINEAR)
    
    img_f = img.tostring()
    
    return img_f


def _int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def get_tfrecord_file(ur_path, test_size, s_width, s_height, *x_filename,):

    #NOTE:data is a tuple (list) of paths of each image

    #NOTE: data can come from the dev-set or test-set

    #NOTE: Same for labeled

    test_img, test_label, train_img, train_label = read_paths_treat(ur_path, test_size)

    file_path = x_filename[0]

    #Open a TFRecordWriter

    writer = tf.python_io.TFRecordWriter(file_path)

    for i in range(len(train_img)):

        new_img = load_image(train_img[i], s_width, s_height)

        new_label = train_label[i]

        new_feature = { 'height': _int64_feature(s_height),
                        'width' : _int64_feature(s_width),
                        'label': _int64_feature(new_label),
                        'image': _bytes_feature(tf.compat.as_bytes(new_img))
        }
        #NOTE: PLease refer to this url for defenirtion: https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto

        example = tf.train.Example(features=tf.train.Features(feature=new_feature))

        writer.write(example.SerializeToString())
    writer.close()

    file_path = x_filename[1]
    writer = tf.python_io.TFRecordWriter(file_path)

    for j in range(len(test_img)):

        new_img = load_image(test_img[j], s_width, s_height)

        new_label = test_label[j]

        new_feature = { 'height': _int64_feature(s_height),
                        'width' : _int64_feature(s_width),
                        'label': _int64_feature(new_label),
                        'image': _bytes_feature(tf.compat.as_bytes(new_img))
        }
        #NOTE: PLease refer to this url for def: https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto

        example = tf.train.Example(features=tf.train.Features(feature=new_feature))

        writer.write(example.SerializeToString())

    writer.close()

    sys.stdout.flush()

    return 0

get_tfrecord_file(UR_DATA_PATH, UR_TEST_SIZE, 256, 256, "tftrain.tfrecords", "tftest.tfrecords")


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
        print( height, width, label, img_string)
    return 0

'''read_file("tfexample.tfrecords")'''