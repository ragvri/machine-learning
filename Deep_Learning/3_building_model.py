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

# using MNIST dataset, set of written examples of handwritten digits, 28 by 28 pixels
mnist = input_data.read_data_sets('/home/raghav/Desktop/Data', one_hot=True)  # 1 is on and rest is off. Could be

# usefull in multiclass
# eg here 10 classes - 0 to 9
# one hot means output of 0  is by [1,0,0,0,0,0,0,0,0]
#               output of 1 is output of 0  is by [0,1,0,0,0,0,0,0,0]

# defining our model

n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500

n_classes = 10
batch_size = 100  # goes through batches of 100 of features and feed them to network at a time and manipulate the
# weights and then another batch and so on

# matrix is height by width
x = tf.placeholder('float', [None, 28 * 28])  # flattening out the matrix
y = tf.placeholder('float')


# x is the data, y is output

def neural_network_model(data):
    # weights are tf variable where the variable is a tf random_normal and we specify the shape of the normal
    # for eg in the hidden_1_layer we have 28*28 inputs and n_nodes_h1 nodes. So a total of 28*28*n_nodes_h1 weights

    hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal([28 * 28, n_nodes_h1])),
                      'biases': tf.constant(0.1, shape=[n_nodes_h1])}

    hidden_2_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_h1, n_nodes_h2])),
                      'biases': tf.constant(0.1, shape=[n_nodes_h2])}

    hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_h2, n_nodes_h3])),
                      'biases': tf.constant(0.1, shape=[n_nodes_h2])}

    output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_h3, n_classes])),
                    'biases': tf.constant(0.1, shape=[n_classes])}

    # (input*weights + bias)
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)  # threshold function

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))  # calculates the
    # diff of prediction
    # to known label
    # minimise the cost

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples) // batch_size):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of ', hm_epochs, ' loss ', epoch_loss)

        # testing
        correct = tf.equal(tf.argmax(prediction, 1),
                           tf.argmax(y, 1))  # argmax returns the index of max value in the arrays

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
