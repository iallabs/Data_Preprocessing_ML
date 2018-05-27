import tensorflow as tf

T_HEIGHT = 256
T_WIDTH = 256

TRAIN_FILE = 'tftrain.tfrecords'
TEST_FILE = 'tftest.tfrecords'

learning_rate = 0.001
num_steps = 43
batch_size = 16
display_step = 10

n_hidden_1 = 1 # 1st layer number of neurons
n_hidden_2 = 1 # 2nd layer number of neurons
num_input = 65536 # MNIST data input (img shape: 256*256)
num_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

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
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
####

def tfrec_data_input_fn(filenames, num_epochs=1, batch_size=16):
    
    def _input_fn():
        def _parse_record(tf_record):
            features = {
                'image': tf.FixedLenFeature([], dtype=tf.string),
                'label': tf.FixedLenFeature([], dtype=tf.int64)
            }
            record = tf.parse_single_example(tf_record, features)

            image_raw = tf.decode_raw(record['image'], tf.uint8)
            

            label = tf.cast(record['label'], tf.int32)
            label = tf.one_hot(label, depth=2)
            

            return { 'image': image_raw }, label
        
        # For TF dataset blog post, see https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_record)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels
    
    return _input_fn

tfrec_dev_input_fn = tfrec_data_input_fn(["tftrain.tfrecords"])
features, labels = tfrec_dev_input_fn()

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
    tfrec_dev_input_fn = tfrec_data_input_fn(["tftest.tfrecords"], batch_size=100)
    features, labels = tfrec_dev_input_fn()
    img, label = sess.run([features['image'], labels])
    print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: img,
                                      Y: label}))
    print(sess.run([weights['h1'], biases['b1'], weights['h2'], biases['b2']]))
    print(sess.run([weights['out'],biases['out']]))
    