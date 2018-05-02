import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt

T_HEIGHT = 28
T_WIDTH = 28
T_Channnels = 1
TRAIN_FILE = 'tftrain.tfrecords'
TEST_FILE = 'tftest.tfrecords'

learning_rate = 0.1
lambda_r = 0.01
num_steps = 1500
num_epoch = 3
batch_size = 16
display_step = 10

n_hidden_1 = 160  # 1st layer number of neurons
n_hidden_2 = 160 # 2nd layer number of neurons

num_input = T_Channnels*T_HEIGHT*T_WIDTH #  data input (img shape: 64*64)
num_classes = 10 # 

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder(tf.int64, [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights['h1'])
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights['h2'])
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights['out'])
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    z1 = tf.nn.tanh(layer_1)

    layer_2 = tf.add(tf.matmul(z1, weights['h2']), biases['b2'])

    z2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(z2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer

loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)
regul = tf.contrib.layers.l2_regularizer(scale=0.1)
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regul, reg_variables) #Regularization term
loss_op = tf.reduce_mean(loss_op + reg_term)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
####


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    # Run the initializer
    '''img, label = sess.run([features['image'], labels])'''
    batch = mnist.train.next_batch(batch_size)
    print(batch[0][1].shape)
    for step in range(1,1+num_steps):
        
        '''for i in range(16):
            print(label[i])
            plt.imshow(img[i].reshape(256,256))
            plt.show()'''
        # Run optimization op (backprop)
        
        '''sess.run(train_op, feed_dict={X: batch[0], Y: batch[1]})

        if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch[0],
                                                                Y: batch[1]})
            print("Step " + str(step) + ", Minibatch Loss= " + \
            "{:.4f}".format(loss) + ", Training Accuracy= " + \
            "{:.3f}".format(acc))
    print("Optimization finished")
    
    
    
    print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
    
    
    tfrec_dcatvdog= tfrec_data_catvdog(["tfpred.tfrecords"])
    a = tfrec_dcatvdog()
    for i in range(750):
        img = sess.run([a['image']])
        print("prediction :", \
                sess.run(prediction, feed_dict={X:img}))
    
    print(sess.run([weights['h1'], biases['b1'], weights['h2'], biases['b2']]))
    print(sess.run([weights['out'],biases['out']]))'''