from image_pro.tfRecord.decode import tfrec_data_input_fn, tfrec_data_catvdog
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from image_pro.parameters import *
import argparse
import os

num_classes = 2
num_step = 1500
num_epochs = 10
learning_rate = 0.001
beta = 0.0025

nets = {"64":[(3,3,3,64),(3,3,64,64)], 
        "128":[(3,3,64,128), (3,3,128,128)], 
        "256":[(3,3,128,256), (3,3,256,256),(3,3,256,256)],
        "512-1":[(3,3,256,512), (3,3,512,512), (3,3,512,512)],
        "512-2": [(3,3,512,512),(3,3,512,512),(3,3,512,512)]}

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
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

    return X, Y

def initialize_parameters(spec_dict):
    parameters = {}
    spec = 0
    with tf.device('/cpu:0'):
        for _, value in spec_dict.items():
            print(value)
            for i in range(len(value)):
                with tf.name_scope('weights'):
                    W = tf.get_variable("W"+str(spec)+str(i),[value[i][0],value[i][1],value[i][2],value[i][3]], initializer=tf.random_normal_initializer(seed=0))
                    variable_summaries(W)
                parameters["W"+str(spec)+str(i)] = W

            spec += 1
    return parameters


def vgg_conv(list_X, W, W_bis, cnv_strides=1, mp_size=2, mp_strides=2, cnv_padd='SAME'):
    
    with tf.name_scope('Conv_Z'):
        Z = tf.nn.conv2d(list_X, W, strides = [1,cnv_strides,cnv_strides,1], padding = cnv_padd)
        tf.summary.histogram('Convolution', Z)
    # RELU
    with tf.name_scope('Second_Conv_Z'):
        Z_bis = tf.nn.conv2d(Z, W_bis, strides = [1,cnv_strides,cnv_strides,1], padding = cnv_padd)
        tf.summary.histogram('Convolution_bis', Z_bis)
    return Z_bis


def activate(pre_act, cnv_strides=1, mp_size=2, mp_strides=2):
    
    with tf.name_scope('RElu_A'):
        A = tf.nn.relu(pre_act)
        tf.summary.histogram('Relu', A)

        # MAXPOOL: window max_p_size,max_p_size,  max_p_strides, padding 'VALID'
    with tf.name_scope('Max_pool'):
        P = tf.nn.max_pool(A, ksize = [1,mp_size,mp_size,1], strides = [1,mp_strides,mp_strides,1], padding = 'VALID')
        tf.summary.histogram('max_pool', P)
        
    return P



def vgg_sub_conv(Z, W = tf.constant(0), cnv_strides=1, mp_size=2, mp_strides=2, conv=True):
    if conv:
        with tf.name_scope("sub_conv"):
            alpha = tf.nn.conv2d(Z, W, strides = [1,cnv_strides,cnv_strides,1], padding = 'SAME')
            tf.summary.histogram('Sub_Convolution', alpha)
        P = activate(alpha)
    else:
        P = activate(Z)
    return P


def vgg_block(X, parameters):

    W00 = parameters['W00']
    W01 = parameters['W01']
    W10 = parameters['W10']
    W11 = parameters['W11']
    W20 = parameters['W20']
    W21 = parameters['W21']
    W22 = parameters['W22']
    W30 = parameters['W30']
    W31 = parameters['W31']
    W32 = parameters['W32']
    W40 = parameters['W40']
    W41 = parameters['W41']
    W42 = parameters['W42']


    with tf.name_scope("1ere_couche"):
        P1 = vgg_conv(X, W00, W01, cnv_strides=1)
        P11 = vgg_sub_conv(P1, conv=False)
        variable_summaries(P11)

    with tf.name_scope("2eme_couche"):
        P2 = vgg_conv(P11, W10, W11, cnv_strides=1)
        P12 = vgg_sub_conv(P2, conv=False)
        variable_summaries(P12)

    with tf.name_scope("3eme_couche"):
        P3 = vgg_conv(P12, W20,W21, cnv_strides=1)
        P13 = vgg_sub_conv(P3, W=W22, conv=True)
        variable_summaries(P13)

    with tf.name_scope("4eme_couche"):
        P4 = vgg_conv(P13, W30,W31, cnv_strides=1)
        P14 = vgg_sub_conv(P4, W=W32, conv=True)
        variable_summaries(P14)

    with tf.name_scope("5eme_couche"):
        P5 = vgg_conv(P14, W40, W41, cnv_strides=1)
        P14 = vgg_sub_conv(P5, W=W42, conv=True)
        variable_summaries(P14)

    f = tf.contrib.layers.flatten(P14)

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




iterator = tfrec_data_input_fn("tftrain.tfrecords", num_epochs=num_epochs)
alp = iterator()

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
X, Y = create_placeholders(T_HEIGHT, T_WIDTH, T_Channnels, num_classes)

# Initialize parameters
parameters = initialize_parameters(nets)
    
# Forward propagation: Build the forward propagation in the tensorflow graph
Z3 = vgg_block(X, parameters)

    
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
with tf.Session() as sess:
    costs = []
    acc_train = []
    train_writer = tf.summary.FileWriter(os.getcwd()+'/log_train_vgg', sess.graph)
    # Run the initialization
    sess.run(init)
    sess.run(alp.initializer)
    images, label = alp.get_next()
    # Do the training loop
    for epoch in range(num_epochs):
        train_accuracy = 0.
        train_cost = 0.
        # Select a minibatch
            
        
        for i in range(num_step):
            image, label_train = sess.run([images['image'], label])
            img = image.reshape(16, T_HEIGHT,T_WIDTH,T_Channnels)
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            summary, temp_cost = sess.run([merged, [optimizer, cost, accuracy]], feed_dict={X:img, Y:label_train})
            train_writer.add_summary(summary, global_step=float(epoch+(i/num_step)))
            temp_acc = temp_cost[2]
            train_cost += temp_cost[1]/num_step
            train_accuracy += temp_acc/num_step        
           
        if epoch % 1 == 0:
            acc_train.append(train_accuracy)
            print ("train after epoch %i: %f" % (epoch, train_accuracy))
            costs.append(train_cost)
            print ("cost after epoch %i: %f" % (epoch, train_cost))

    train_writer.close()
    iterator_test = tfrec_data_input_fn("tftrain.tfrecords", batch_size = 20)
    test = iterator_test()
    sess.run(test.initializer)
    X_test, Y_test = test.get_next()
    test_accuracy = 0.0
        # plot the cost
    for i in range(99):
        X_test_b, Y_test_b = sess.run([X_test['image'], Y_test])
        img_test = X_test_b.reshape(20, T_HEIGHT,T_WIDTH,1)
        temp_test_accuracy = accuracy.eval({X: img_test, Y: Y_test_b})
        test_accuracy += temp_test_accuracy/100
    print("Test Accuracy:", test_accuracy)
    input_pred = tfrec_data_catvdog("tfpred.tfrecords")
    image_p = input_pred()
    image_pred = image_p.get_next()
    for i in range(10):
        a = sess.run(image_pred['image'])
        pred = sess.run(predict_op, feed_dict={X:a.reshape(-1, T_HEIGHT,T_WIDTH,T_Channnels)})
        print(pred)
    sess.close()