from random import shuffle, seed
import os
import glob

import numpy as np

import cv2

import tensorflow as tf

import sys




UR_DATA_PATH = "./train/*/*.jpg"
UR_PRED_PATH = "./test1/test1/*.jpg"

UR_TEST_SIZE = 1000




# This function returns the different paths and corresponding labels
def shuffling_data(data_path, labels):
    c = list(zip(data_path, labels))

    shuffle(c)

    #NOTE: data and labeld are tuples. Care about immutability 

    data, labeled = zip(*c) #NOTE:ith element in data correpand to ith label elmt

    return data, labeled

def read_paths_treat(ur_path, test_size):
    pathname = os.path.dirname(ur_path)
    #Reading the path of each data(ex: image) and then extracting the labels from path
    data_path = glob.glob(ur_path)
    '''labels = [int(i.split('.')[0][-1]) for i in data_path]'''#xray
    labels = [i.split('\\')[-1] for i in data_path]  #Cat vs Dog
    labels_num = [0 if "cat" in i else 1 for i in labels] #Cat vs Dog
    len_d = len(labels_num)
    k = int(len_d/2)
    a = int((len_d-test_size)/2)
    b = int(len_d-(test_size/2))
    data_path_test, labels_test = data_path[a : k], labels_num[a:k]
    data_path_test = data_path_test + data_path[b:]
    labels_test = labels_test + labels_num[b:]
    data_path_train,labels_train = (data_path[0:a]), (labels_num[0:a])
    data_path_train = data_path_train + data_path[k:b]
    labels_train = labels_train + labels_num[k:b]
    data_path_test, labels_test = shuffling_data(data_path_test, labels_test)
    data_path_train, labels_train = shuffling_data(data_path_train, labels_train)
    return data_path_test, labels_test,data_path_train, labels_train


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

    img_load = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
   
    '''img_after = cv2.cvtColor(img_load, cv2.COLOR_BGR2GRAY)'''
    
    img= cv2.resize(img_load, dsize=(s_width, s_height), interpolation=cv2.INTER_CUBIC)

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
    dog, cat= 0,0
    for i in range(len(train_img)):
        if "cat" in train_img[i]: 
            cat +=1
        else:
            dog +=1
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
    print(dog,cat)
    writer.close()
    sys.stdout.flush()
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


def get_tfrecord_pred(ur_path, s_width, s_height, x_filename,):
    pathname = os.path.dirname(ur_path)
    data_path = glob.glob(ur_path)
    file_path = x_filename
    writer = tf.python_io.TFRecordWriter(file_path)
    for i in range(len(data_path)):
        new_img = load_image(data_path[i], s_width, s_height)
        new_feature = { 'height': _int64_feature(s_height),
                        'width' : _int64_feature(s_width),
                        'image': _bytes_feature(tf.compat.as_bytes(new_img))
                        }
        example = tf.train.Example(features=tf.train.Features(feature=new_feature))
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()
    return 0


'''get_tfrecord_pred(UR_PRED_PATH, 256, 256, "tfpred.tfrecords")
''''''get_tfrecord_file(UR_DATA_PATH, UR_TEST_SIZE, 256, 256, "tftrain.tfrecords", "tftest.tfrecords")'''