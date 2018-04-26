from random import shuffle, seed
import glob
import numpy as np
import cv2
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import logging

print("Starting steps")
logging.log(0, "hello fker")

learning_rate = 0.001

num_steps = 42

batch_size = 16

display_step = 10



# Network Parameters

num_input = 4096 # MNIST data input (img shape: 64*64)

num_classes = 2 # MNIST total classes (0-1 digits)

dropout = 0.75 # Dropout, probability to keep units



# tf Graph input

X = tf.placeholder(tf.float32, [None, num_input])

Y = tf.placeholder(tf.float32, [None, num_classes])

keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)





# Create some wrappers for simplicity

def conv2d(x, W, b, strides=1):

    # Conv2D wrapper, with bias and relu activation

    x = tf.nn.conv2d(x, W, strides=[3, strides, strides, 3], padding='SAME')

    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)





def maxpool2d(x, k=2):

    # MaxPool2D wrapper

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],

                          padding='SAME')





# Create model

def conv_net(x, weights, biases, dropout):

    # Xray data input is a 1-D vector of 784 features (28*28 pixels)

    # Reshape to match picture format [Height x Width x Channel]

    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]

    x = tf.reshape(x, shape=[-1, 64, 64, 3])



    # Convolution Layer

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    # Max Pooling (down-sampling)

    conv1 = maxpool2d(conv1, k=2)



    # Convolution Layer

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Max Pooling (down-sampling)

    conv2 = maxpool2d(conv2, k=2)



    # Fully connected layer

    # Reshape conv2 output to fit fully connected layer input

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

    fc1 = tf.nn.relu(fc1)

    # Apply Dropout

    fc1 = tf.nn.dropout(fc1, dropout)



    # Output, class prediction

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out

def read_file(x_filename, x_capacity=662):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
                'train/label': tf.FixedLenFeature([], tf.int64)
    }
    queue = tf.train.string_input_producer([x_filename], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue)
    features = tf.parse_single_example(serialized_example, features=feature)
    
    #x means training, dev or test
    x_image = tf.decode_raw(features['train/image'], tf.float32)
    x_label = tf.cast(features['train/label'], tf.int64)

    x_image = tf.reshape(x_image,[64,64,3])
    #NOTE: In this line, we can add a instruction to preprocess the images to their original shape
    return x_image, x_label

# Store layers weight & bias
print("Starting cccccc")
weights = {

    # 5x5 conv, 1 input, 32 outputs

    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),

    # 5x5 conv, 32 inputs, 64 outputs

    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),

    # fully connected, 7*7*64 inputs, 1024 outputs

    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),

    # 1024 inputs, 2 outputs (class prediction)

    'out': tf.Variable(tf.random_normal([1024, num_classes]))

}



biases = {

    'bc1': tf.Variable(tf.random_normal([32])),

    'bc2': tf.Variable(tf.random_normal([64])),

    'bd1': tf.Variable(tf.random_normal([1024])),

    'out': tf.Variable(tf.random_normal([num_classes]))

}



# Construct model

logits = conv_net(X, weights, biases, keep_prob)

prediction = tf.nn.softmax(logits)



# Define loss and optimizer

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(

    logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)





# Evaluate model
print("Starting bbbbbb")
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# Initialize the variables (i.e. assign their default value)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



# Start training

with tf.Session() as sess:
    # Run the initializer
    print("Starting aaaaaa")
    sess.run(init_op)
    #Passing the fuc to extract data
    x_train, y_train = read_file("./Tfexamp.tfrecords")
    #Applying sess.run() turn the data to numpy ndarray
    x_train = sess.run(x_train)
    y_train = sess.run(y_train)
    print("Starting steps")
    x_train, y_train = tf.train.shuffle_batch([x_train,y_train],
                                                batch_size=batch_size, num_threads=1,
                                                capacity=662, min_after_dequeue=1,
                                                allow_smaller_final_batch=True)
    for step in range(1, num_steps+1):

        batch_x, batch_y = x_train, y_train

        # Run optimization op (backprop)

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})

        if step % display_step == 0 or step == 1:

            # Calculate batch loss and accuracy

            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,

                                                                 Y: batch_y,

                                                                 keep_prob: 1.0})

            print("Step " + str(step) + ", Minibatch Loss= " + \

                  "{:.4f}".format(loss) + ", Training Accuracy= " + \

                  "{:.3f}".format(acc))


        x_train.next()
        y_train.next()
    print("Optimization Finished!")

    sess.close()

    '''# Calculate accuracy for 256 MNIST test images

    print("Testing Accuracy:", \

        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],

                                      Y: mnist.test.labels[:256],

                                      keep_prob: 1.0}))'''