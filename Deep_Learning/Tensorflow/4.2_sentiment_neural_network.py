import tensorflow as tf
import pickle
import numpy as np

f = open("sentiment_set.pickle", 'rb')
data_pickle = pickle.load(f)
train_x, train_y, test_x, test_y = data_pickle

n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500

n_classes = 2
batch_size = 100
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')


# x is the data, y is output

def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_h1])),
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
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch+1, 'completed out of ', hm_epochs, ' loss ', epoch_loss)

        # testing
        correct = tf.equal(tf.argmax(prediction, 1),
                           tf.argmax(y, 1))  # argmax returns the index of max value in the arrays

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy ', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)
