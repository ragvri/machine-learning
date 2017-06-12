"""
Convolution Neural Network: State of Art,: Used in images

We have input data. Do some convolutions which creates feature maps. Then do some pooling. Conv+Pooling is the hidden
layer. Then a fully connected layer added. Then output.

input -> (convolution -> pooling) -> fully connected layer(hidden layers of simple neural nets) -> output
                  |
                 \ /
                hidden layer

Convolve :  Creating feature maps from original dataset. eg let the dataset be image of a cat. take a 3*3 window. \
This window can be moved over the image. The creation of these new windows which have some value in them is called
convolution. We can now instead of image, create a feature map(2*2)grid which has the values of the windows.

Pooling: eg 3*3 pooling i.e 3*3 window. Pooling is simplifying the window. For eg max pool. Take the max value out of
the 3*3 pool

"""
"""
Our model
input data > Weight it > hidden layer 1 (activation function) > weights > Hidden Layer 2(activation function)> weights
> output layer.

In a neural network, this data is passed straight through. That passing of data is called feed forward

Compare output to intended output.> cost function

optimisation function(optimiser) which will minimise the cost eg(Adam Optimiser, AdaGrad)
This optimiser goes backwards and manipulates the weights.
This motion is called Backward Propogation

feed forward + backpropogation = epoch-> Cycle. Cost minimised at each cycle

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # window is 2*2, and will move
    # 2 pixels


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),  # we are taking 5*5 window on original image
               # and taking 32 outputs
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
