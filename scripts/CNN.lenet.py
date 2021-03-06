from image_pro.tfRecord.decode import tfrec_data_input_fn, tfrec_data_catvdog
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from image_pro.parameters import *


num_classes = 2
num_step = 1500
num_epochs = 40
learning_rate = 0.0001
beta = 0.0025
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
    X = tf.placeholder(tf.float32,shape=[None,n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32,shape=[None, n_y])
    
    return X, Y

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:

    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours

    W1 = tf.get_variable("W1", [5,5,1,32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5,5,32,64], initializer=tf.contrib.layers.xavier_initializer(seed=0))


    b1 = tf.get_variable("b1", [1,1,1,32],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b2 = tf.get_variable("b2", [1,1,1,64],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
                  }
    
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

  

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME') + b1
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 4,4 sride 1, padding 'VALID'
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME') + b2
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 2, padding 'VALID'
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    # FLATTEN
    P5 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z6 = tf.contrib.layers.fully_connected(P5, num_classes, activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer, biases_regularizer=tf.contrib.layers.l2_regularizer)

    return Z6

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z3, labels = Y))
    
    return cost


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
parameters = initialize_parameters()
    
# Forward propagation: Build the forward propagation in the tensorflow graph
Z3 = forward_propagation(X, parameters)

    
# Cost function: Add cost function to tensorflow graph
cost = compute_cost(Z3, Y)
regularizers = tf.nn.l2_loss(parameters["W1"]) + tf.nn.l2_loss(parameters["W2"]) + \
                   tf.nn.l2_loss(parameters["b1"]) + tf.nn.l2_loss(parameters["b2"])
cost = tf.reduce_mean(cost+beta*regularizers)
# Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate the correct predictions
predict_op = tf.argmax(Z3, 1)
correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
# Calculate accuracy on the test set
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Initialize all the variables globally
init = tf.global_variables_initializer()
     
# Start the session to compute the tensorflow graph

with tf.Session() as sess:
    costs = []
    acc_train = []
    # Run the initialization
    sess.run(init)
    sess.run(alp.initializer)
    images, label = alp.get_next()
    # Do the training loop
    for epoch in range(num_epochs):
        train_accuracy = 0.
        minibatch_cost = 0.
        # Select a minibatch
            
        
        for i in range(num_step):
            image, label_train = sess.run([images['image'], label])
            img = image.reshape(16, T_HEIGHT,T_WIDTH,T_Channnels)
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: img, Y: label_train})
            temp_acc = accuracy.eval({X:img, Y: label_train})
            train_accuracy += temp_acc/num_step
            minibatch_cost += temp_cost / num_step              
           
        if epoch % 1 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            costs.append(minibatch_cost)
            acc_train.append(train_accuracy)
            print ("train after epoch %i: %f" % (epoch, train_accuracy))
        
        # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    iterator_test = tfrec_data_input_fn("tftest.tfrecords", batch_size = 20)
    test = iterator_test()
    sess.run(test.initializer)
    c, d = test.get_next()
    test_accuracy = 0.0
    for i in range(49):
        X_test, Y_test = sess.run([c['image'], d])
        img_test = X_test.reshape(20, T_HEIGHT,T_WIDTH,T_Channnels)
        test_accuracy += accuracy.eval({X: img_test, Y: Y_test})/50
    print("Test Accuracy:", test_accuracy)
    input_pred = tfrec_data_catvdog("tfpred.tfrecords")
    image_p = input_pred()
    image_pred = image_p.get_next()
    for i in range(35):
        a = sess.run(image_pred['image'])
        pred = sess.run(predict_op, feed_dict={X:a.reshape(-1, T_HEIGHT,T_WIDTH,T_Channnels)})
        print(pred)
    sess.close()