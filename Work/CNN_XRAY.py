import tensorflow as tf
import tfRecord as trec


T_HEIGHT = 32
T_WIDTH = 32
T_channels = 3

TRAIN_FILE = 'tftrain.tfrecords'
TEST_FILE = 'tftest.tfrecords'

learning_rate = 0.001

num_steps = 1500

batch_size = 8

display_step = 10



# Network Parameters

num_input = T_HEIGHT * T_WIDTH * T_channels

num_classes = 2 




# tf Graph input

X = tf.placeholder(tf.float32, [None, num_input])

Y = tf.placeholder(tf.int64, [None, num_classes])







# Create some wrappers for simplicity

def conv2d(x, W, b, strides=1):

    # Conv2D wrapper, with bias and relu activation

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)





def maxpool2d(x, k=2):

    # MaxPool2D wrapper

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],

                          padding='SAME')





# Create model

def conv_net(x, weights, biases):


    x = tf.reshape(x, shape=[-1, T_HEIGHT, T_WIDTH, T_channels])



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

    # Output, class prediction

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out



# Store layers weight & bias

weights = {

    # 5x5 conv, 1 input, 32 outputs

    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),

    # 5x5 conv, 32 inputs, 64 outputs

    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),

    # fully connected, 32*32*64 inputs, 1024 outputs

    'wd1': tf.Variable(tf.random_normal([8*8*64, 512])),

    # 1024 inputs, 10 outputs (class prediction)

    'out': tf.Variable(tf.random_normal([512, num_classes]))

}



biases = {

    'bc1': tf.Variable(tf.random_normal([32])),

    'bc2': tf.Variable(tf.random_normal([64])),

    'bd1': tf.Variable(tf.random_normal([512])),

    'out': tf.Variable(tf.random_normal([num_classes]))

}



# Construct model

logits = conv_net(X, weights, biases)

prediction = tf.nn.softmax(logits)



# Define loss and optimizer

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                         logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)





# Evaluate model

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.int64))



# Initialize the variables (i.e. assign their default value)
tfrec_dev_input_fn = trec.decode_tfrecord.tfrec_data_input_fn(["catvdog.tfrecords"],batch_size=batch_size)
features, labels = tfrec_dev_input_fn()

init = tf.global_variables_initializer()



# Start training

with tf.Session() as sess:



    # Run the initializer

    sess.run(init)



    for step in range(1, num_steps+1):
        img, label = sess.run([features['image'], labels])
        
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: img, Y: label})
        
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: img,
                                                                 Y: label})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    '''tfrec_dev_input_fn = trec.decode_tfrecord.tfrec_data_input_fn(["catvdogtest.tfrecords"], batch_size=1000)
    features, labels = tfrec_dev_input_fn()
    img, label = sess.run([features['image'], labels])
    print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: img,
                                      Y: label}))'''
    sess.close()