import tensorflow as tf
import numpy as np


# a = tf.constant(2, name='a')
# b = tf.constant(3, name='b')
# c = tf.add(a, b, name='add')
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter('logs/', sess.graph)
#     print(sess.run(a))

# X*W + b

featureSize = 784
hiddenNodeSize = 200

# create the input placeholder
X = tf.placeholder(tf.float32, shape=[None, featureSize], name='X')
weight_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
W = tf.get_variable(name='weight', dtype=tf.float32, shape=[featureSize, hiddenNodeSize], initializer=weight_initializer)

bias_initializer = tf.constant(0., shape=[hiddenNodeSize], dtype=tf.float32)
b = tf.get_variable(name='bias', dtype=tf.float32, initializer=bias_initializer)

x_w = tf.matmul(X, W, name='X_W')
x_w_b = tf.add(x_w, b, name='X_W_b')
# activation function ReLU
h = tf.nn.relu(x_w_b, name='ReLU')

# to initialize variables
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    d = {
        X: np.random.rand(100, featureSize)
    }
    writer = tf.summary.FileWriter('logs/', sess.graph)
    result = sess.run(h, feed_dict=d)
    print(result)
    # don't forget to close
    writer.close()
