import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from neuro_net import *
import numpy as np

def reshape(x):
    img_size = int(np.sqrt(x.shape[-1]))
    num_ch = 1
    dataset = x.reshape((-1, img_size, img_size, num_ch)).astype(np.float32)
    return dataset

def randomize(x, y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation,:, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start: end]
    y_batch = y[start: end]
    return x_batch, y_batch


img_h = img_w = 28
num_classes = 10
num_channels = 1

filter_size1 = 5
num_filters1 = 16
stride1 = 1
filter_size2 = 5
num_filters2 = 32
stride2 = 1

hidden_nodes = 200
learning_rate = 0.001

X = tf.placeholder(tf.float32, shape=[None, img_h, img_w, num_channels], name='X')
Y = tf.placeholder(tf.float32, shape=[None, num_classes], name='Y')

conv1 = add_conv_layer(X, filter_size1, num_filters1, stride1, 'conv1')
pool1 = add_pooling_layer(conv1, ksize=2, stride=2, name='pool1')

conv2 = add_conv_layer(pool1, filter_size2, num_filters2, stride2, 'conv2')
pool2 = add_pooling_layer(conv2, ksize=2, stride=2, name='pool2')

flatten = add_flatten_layer(pool2)
hidden_layer = add_fc_layer(flatten, hidden_nodes, 'hidden_layer', tf.nn.relu)
h = add_fc_layer(hidden_layer, num_classes, 'output')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h, labels=Y), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam-op").minimize(loss)
correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(Y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

init = tf.global_variables_initializer()

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x_train, y_train = mnist.train.images, mnist.train.labels
x_train = reshape(x_train)

epochs = 10
batch_size = 100


sess = tf.InteractiveSession()
sess.run(init)

# feed_dict = {
#     X: x_train[0:10],
#     Y: y_train[0:10]
# }
# result = sess.run(conv1, feed_dict=feed_dict)
# print(result.shape)
# result = sess.run(pool1, feed_dict=feed_dict)
# print(result.shape)
# result = sess.run(conv2, feed_dict=feed_dict)
# print(result.shape)
# result = sess.run(pool2, feed_dict=feed_dict)
# print(result.shape)
#
# result = sess.run(flatten, feed_dict=feed_dict)
# print(result.shape)

num_tr_iter = int(len(y_train)/batch_size)
x_valid, y_valid = mnist.validation.images, mnist.validation.labels
x_valid = reshape(x_valid)

for epoch in range(epochs):
    x_train, y_train = randomize(x_train, y_train)
    for i in range(num_tr_iter):
        start = i * batch_size
        end = start + batch_size
        x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
        feed_dict_batch = {
            X: x_batch,
            Y: y_batch
        }
        sess.run(optimizer, feed_dict=feed_dict_batch)
        # if i % 100 == 0:
        #     loss_r, acc_r = sess.run([loss, accuracy], feed_dict=feed_dict_batch)
        #     print(loss_r)
        #     print(acc_r)

    feed_dict_valid = {X: x_valid, Y: y_valid}
    loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
    print(loss_valid)
    print(acc_valid)
sess.close()



