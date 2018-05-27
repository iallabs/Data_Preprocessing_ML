from image_pro.tfRecord.decode import tfrec_data_input_fn, tfrec_data_catvdog
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from image_pro.parameters import *
import argparse
import os
"""import cv2"""

num_classes = 2
num_step = 1500
num_epochs = 10
learning_rate = 0.000001
beta = 0.0025

nets = [(11,11,3,96), (5,5,96,256), (3,3,256,384), (3,3,384,384),
        (3,3,384,256)]

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    with tf.device('/cpu:0'):
        with tf.name_scope('input'):
            X = tf.placeholder(tf.float32,shape=[None,n_H0, n_W0, n_C0], name='x-input')
            variable_summaries(X)
            Y = tf.placeholder(tf.float32,shape=[None, n_y], name='y-input')
            variable_summaries(Y)
    print("aaaaaaa")
    return X, Y

def initialize_parameters(spec_list):
    parameters = {}
    with tf.device('/cpu:0'):
        for spec in range(len(spec_list)):
            with tf.name_scope('weights'):
                W = tf.get_variable("W"+str(spec+1),[spec_list[spec][0],spec_list[spec][1],spec_list[spec][2],spec_list[spec][3]], dtype=tf.float32,initializer=tf.random_normal_initializer(seed=0))
                variable_summaries(W)
            parameters["W"+str(spec+1)] = W
    print("bbbbbb")
    return parameters


def alex_conv(X, W, cnv_strides=1, mp_size=2, mp_strides=2, relu=False, mp=False, cnv_padd='SAME'):
    
    with tf.name_scope('Conv_Z'):
        Z = tf.nn.conv2d(X,W, strides = [1,cnv_strides,cnv_strides,1], padding = cnv_padd)
        tf.summary.histogram('Convolution', Z)
        print(Z)
    # RELU
    if relu:
        with tf.name_scope('RElu_A'):
            A = tf.nn.relu(Z)
            tf.summary.histogram('Relu', A)
    if mp:
        # MAXPOOL: window max_p_size,max_p_size,  max_p_strides, padding 'VALID'
        with tf.name_scope('Max_pool'):
            P = tf.nn.max_pool(A, ksize = [1,mp_size,mp_size,1], strides = [1,mp_strides,mp_strides,1], padding = 'VALID')
            tf.summary.histogram('max_pool', P)
    else:
        P = Z
    return P

def alex_block(X, parameters):

    W1 = parameters['W1']

    W2 = parameters['W2']

    W3 = parameters['W3']

    W4 = parameters['W4']

    W5 = parameters['W5']

    with tf.name_scope("1ere_conv"):
        P1 = alex_conv(X, W1, cnv_strides=4, relu=True, mp=True, cnv_padd='VALID')
        
    with tf.name_scope("2eme_conv"):
        P2 = alex_conv(P1, W2, relu=True, mp=True)
        
    with tf.name_scope("3eme_conv"):
        P3 = alex_conv(P2, W3, relu=False, mp=False)
        
    with tf.name_scope("4eme_conv"):
        P4 = alex_conv(P3, W4, relu=False, mp=False)
        
    with tf.name_scope("5eme_conv"):
        P5 = alex_conv(P4, W5,relu=True, mp=True)
        
    f = tf.contrib.layers.flatten(P5)
    with tf.name_scope("1_er_ffc"):
        f1 = tf.contrib.layers.fully_connected(f, 4096)
        variable_summaries(f1)
    with tf.name_scope("2_eme_ffc"):
        f2 = tf.contrib.layers.fully_connected(f1,4096)
        variable_summaries(f2)
    with tf.name_scope("3_eme_ffc"):
        f3 = tf.contrib.layers.fully_connected(f2, 2)
        variable_summaries(f3)
    
    return f3

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    with tf.name_scope('total'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z3, labels = Y))
        tf.summary.histogram('cross_entropy: cost', cost)
    return cost

def regulize (dict_param):
    with tf.name_scope('regulization'):
        regularizers = 0.0
        for key in dict_param.keys():
            regularizers += tf.nn.l2_loss(dict_param[key])
        variable_summaries(regularizers)
    
    return regularizers




iterator = tfrec_data_input_fn("tftrain.tfrecords", num_epochs=num_epochs, is_training=True)
alp = iterator()
iterator_test = tfrec_data_input_fn("tftest.tfrecords", num_epochs=num_epochs,batch_size = 20, is_training=False)
test = iterator_test()
"""
Implements a three-layer ConvNet in Tensorflow:
CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
Arguments:
X_train -- training set, of shape (None, 256, 256, 1)
Y_train -- test set, of shape (None, n_y = 2)
X_test -- training set, of shape (None, 256, 256, 1)
Y_test -- test set, of shape (None, n_y = 2)
learning_rate -- learning rate of the optimization
num_epochs -- number of epochs of the optimization loop
minibatch_size -- size of a minibatch
print_cost -- True to print the cost every 100 epochs
    
Returns:
train_accuracy -- real number, accuracy on the train set (X_train)
test_accuracy -- real number, testing accuracy on the test set (X_test)
parameters -- parameters learnt by the model. They can then be used to predict.
"""
# Create Placeholders of the correct shape
X, Y = create_placeholders(N_HEIGHT, N_WIDTH, T_Channnels, num_classes)

# Initialize parameters
parameters = initialize_parameters(nets)
    
# Forward propagation: Build the forward propagation in the tensorflow graph
Z3 = alex_block(X, parameters)

    
# Cost function: Add cost function to tensorflow graph
cost = compute_cost(Z3, Y)
"""regula = regulize(parameters)
cost = tf.reduce_mean(cost+beta*regula)"""
# Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate the correct predictions
with tf.name_scope('accuracy'):
    predict_op = tf.argmax(Z3, 1)
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
    with tf.name_scope('accuracy'):    
    # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
# Initialize all the variables globally
init = tf.global_variables_initializer()
     
# Start the session to compute the tensorflow graph
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    costs = []
    acc_train = []
    
    train_writer = tf.summary.FileWriter(os.getcwd()+'/log_train', sess.graph)
    # Run the initialization
    sess.run(init)
    sess.run(alp.initializer)
    sess.run(test.initializer)
    images, label = alp.get_next()
    X_test, Y_test = test.get_next()
    # Do the training loop
    for epoch in range(num_epochs):
        train_accuracy = 0.
        train_cost = 0.
        test_accuracy = 0.
        # Select a minibatch

        
        for i in range(num_step):
            image, label_train = sess.run([images, label])
            """for k in range(16):
                cv2.imshow("a", image[k])
                cv2.waitKey()"""
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            summary, temp_cost = sess.run([merged, [optimizer, cost, accuracy]], feed_dict={X:image, Y:label_train})
            train_writer.add_summary(summary, global_step=float(epoch+(i/num_step)))
            temp_acc = temp_cost[2]
            train_cost += temp_cost[1]/num_step
            train_accuracy += temp_acc/num_step        
           
        if epoch % 1 == 0:
            acc_train.append(train_accuracy)
            print ("train after epoch %i: %f" % (epoch, train_accuracy))
            costs.append(train_cost)
            print ("cost after epoch %i: %f" % (epoch, train_cost))
        if epoch%2 == 0:       
            for i in range(49):
                X_test_b, Y_test_b = sess.run([X_test, Y_test])

                temp_test_accuracy = accuracy.eval({X: X_test_b, Y: Y_test_b})
                test_accuracy += temp_test_accuracy/50
            print("Test Accuracy:", test_accuracy)

    train_writer.close()
        
    input_pred = tfrec_data_catvdog("tfpred.tfrecords")
    image_p = input_pred()
    image_pred = image_p.get_next()
    for i in range(10):
        a = sess.run(image_pred['image'])
        pred = sess.run(predict_op, feed_dict={X:a.reshape(-1, T_HEIGHT,T_WIDTH,T_Channnels)})
        print(pred)
    sess.close()