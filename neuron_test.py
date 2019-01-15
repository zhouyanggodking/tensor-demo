import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from neuro_net import *
import matplotlib.pyplot as plt

# build neural network with three layers of 200 hidden layer nodes
# input_nodes, hidden_nodes, output_nodes

input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# hyper-parameters
learning_rate = 0.0001  # The optimization learning rate
epochs = 30  # Total number of training epochs
batch_size = 100  # Training batch size
display_freq = 100  # Frequency of displaying the training results

X = tf.placeholder(name="X", shape=[None, input_nodes], dtype=tf.float32)
Y = tf.placeholder(name="y", shape=[None, output_nodes], dtype=tf.float32)


# calc hidden layer
h1 = add_fc_layer(X, hidden_nodes, "hidden", tf.nn.relu)

#calc output layer
h2 = add_fc_layer(h1, output_nodes, "output")


h = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h2, name="softmax")

loss = tf.reduce_mean(h)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam-op").minimize(loss)

correct_prediction = tf.equal(tf.argmax(h2, 1), tf.argmax(Y, 1), name="correct_prediction")

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

# read data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

feed_dict = {
    X: mnist.train.images,
    Y: mnist.train.labels
}

# print(sess.run(accuracy, feed_dict=feed_dict))
# while True:
#
#     # loss_r = sess.run(loss, feed_dict=feed_dict)
#     # print(loss_r)
#
#     optimizer_r = sess.run(optimizer, feed_dict=feed_dict)
#
#     # accuracy_r = sess.run(accuracy, feed_dict=feed_dict)
#     # print(accuracy_r)
#
#     loss_r, accuracy_r = sess.run([loss, accuracy], feed_dict=feed_dict)
#     print(loss_r)
#     print(accuracy_r)
#     if accuracy_r > 0.98:
#         break


# stochastic gradient descendent
num_iter = int(mnist.train.num_examples / batch_size)

x_valid, y_valid = mnist.validation.images, mnist.validation.labels
writer = tf.summary.FileWriter('logs/', sess.graph)
for epoch in range(epochs):
    for iteration in range(num_iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict_batch = {
            X: batch_x,
            Y: batch_y
        }
        optimizer_r = sess.run(optimizer, feed_dict=feed_dict_batch)
        # if iteration % 100 == 0:
        #     loss_r, accuracy_r = sess.run([loss, accuracy], feed_dict=feed_dict_batch)
        # print(loss_r)
        # print(accuracy_r)

    feed_dict_valid = {
        X: x_valid,
        Y: y_valid
    }
    loss_valid, accuracy_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
    print(loss_valid)
    print(accuracy_valid)


# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='binary')
# plt.show()

sess.close()
