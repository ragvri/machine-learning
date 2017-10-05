"""
   1) Recurrent Neural Networks: Solves a problem that involves time: eg a machine playing
   catch, it needs to know if the ball is moving away or coming towards. They are used with languages as tense plays a
   role in language.

   LSTM cell (Long Short Term Memory Cell) most common cells used with RNN.

   In deep neural network, we had inputs with weights, which was then fed to a neuron. However order in which they
   were fed did not matter. The inputs were fed to an activation function(neurons) and then we got the output
    In RNN, X1 gets send into the activation function and the output is fed back to the activation function. So at t=0,
    only input fed to activation function. At t=1, both input and output of activation function fed back to activation
    function

    Consider "Raghav drove car" where each word is a feature. In the deep neural network, "Raghav drove car" and "car
    drove Raghav" is same.
"""


"""
Say , u have a 5*5 image and u have 1 such image then it is :

x = np.ones((1,5,5))

so u have , 

x  =  array([[[ 1.,  1.,  1.,  1.,  1.],
              [ 1.,  1.,  1.,  1.,  1.],
              [ 1.,  1.,  1.,  1.,  1.],
              [ 1.,  1.,  1.,  1.,  1.],
              [ 1.,  1.,  1.,  1.,  1.]]])

now for the rnn u need to convert each row of pixel into a single chunk.
so , u would have 5 chunks of 5 values each
so, u need to convert each row to an array

x = np.transpose(x,(1,0,2))

this swaps the 0th dim with the 1st dim . so, u get shape of x as (5,1,5)
which is 5 arrays of 1 chunk each of 5 elements 

x = array([[[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]]])

now , u need to remove 1 pair of extra braces . so flatten by one dimension

x = np.reshape(x,(-1,chunk_size))

so, u will have :

x = array([[ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.]])

and finally u will need to split the entire thing into 5 chunks(5 arrays)
x = np.split(x,n_chunks,0)

so, finally u have :

x = [array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]]), 
array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]])]
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# using MNIST dataset,
mnist = input_data.read_data_sets('/home/raghav/Desktop/Data', one_hot=True)

n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500

hm_epochs = 3
n_classes = 10
batch_size = 128
chuck_size = 28
n_chunks = 28
rnn_size = 128

# images are 28*28
x = tf.placeholder('float', [None, n_chunks, chuck_size])  # flattening out the matrix
y = tf.placeholder('float')


def recurrent_neural_network_model(x):
    layer = {'weights': tf.Variable(tf.truncated_normal([rnn_size, n_classes])),
             'biases': tf.constant(0.1, shape=[n_classes])}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chuck_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))  # calculates the

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples) // batch_size):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chuck_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of ', hm_epochs, ' loss ', epoch_loss)

        # testing
        correct = tf.equal(tf.argmax(prediction, 1),
                           tf.argmax(y, 1))  # argmax returns the index of max value in the arrays

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy ', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chuck_size)),
                                          y: mnist.test.labels}))


train_neural_network(x)
