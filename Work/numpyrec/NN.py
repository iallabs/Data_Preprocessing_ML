import tensorflow as tf
import numpy as np
T_HEIGHT = 256
T_WIDTH = 256

TRAIN_FILE = 'tftrain.tfrecords'
TEST_FILE = 'tftest.tfrecords'

learning_rate = 0.001
num_steps = 1500
batch_size = 16
display_step = 10

n_hidden_1 = 3 # 1st layer number of neurons
n_hidden_2 = 3 # 2nd layer number of neurons
num_input = 49152 #  data input (img shape: 256*256)
num_classes = 2 # 

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder(tf.int64, [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    z1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(z1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    '''z2 = tf.nn.relu(layer_2)'''
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
   labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
####


# Initialize the variables (i.e. assign their default value)
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
    print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: img,
                                      Y: label}))
    for i in range(750):
        img = sess.run([a['image']])
        print("prediction :", \
                sess.run(prediction, feed_dict={X:img}))
    print(sess.run([weights['h1'], biases['b1'], weights['h2'], biases['b2']]))
    print(sess.run([weights['out'],biases['out']]))
    


    