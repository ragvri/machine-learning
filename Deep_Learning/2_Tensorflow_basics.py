"""
Tensor: Array like object
TensorFlow: Has tons of deeplearning functions

In TensorFlow: Define the model in abstract terms. When ready, run the session.

So tensorflow has a computation graph where we model everything.
Then we run the session
"""
import tensorflow as tf

# construct the computation graph, first thing to do
x1 = tf.constant(5)
x2 = tf.constant(6)

# result = x1*x2 # can do this but not efficient
result = tf.mul(x1, x2)
print(result)
# result is a tensor object

# to actually get the answer we need to run it in a session
# Method 1
# sess = tf.Session()
# print(sess.run(result))
# sess.close()

# Method 2, better
with tf.Session() as sess:
    print(sess.run(result))
