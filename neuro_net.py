import tensorflow as tf

# build neural network with three layers of 200 hidden layer nodes
# input_nodes, hidden_nodes, output_nodes


def init_weight(shape):
    w_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    return tf.get_variable(name='w', shape=shape, initializer=w_initializer)
    # w_initializer = tf.constant(1., shape=[input_nodes, output_nodes])
    # return tf.get_variable(name='w_' + name, initializer=w_initializer)


def init_bais(shape):
    b_initializer = tf.zeros(shape=shape)
    return tf.get_variable(name='b', initializer=b_initializer)


def add_fc_layer(x, output_nodes, name, activation=None):
    with tf.variable_scope(name):
        input_nodes = x.shape[1]
        w = init_weight([input_nodes, output_nodes])
        b = init_bais([output_nodes])
        h = tf.matmul(x, w)
        h += b
        if activation is None:
            return h
        else:
            return activation(h)


# x shape = (sample_size, img_with, img_height, num_channels)
def add_conv_layer(x, filter_size, num_filters, stride, name):
    with tf.variable_scope(name):
        num_in_channel = x.shape[-1]
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        W = init_weight(shape=shape)
        b = init_bais(shape=[num_filters])
        # x = (sample_size, img_with, img_height, num_channels)
        # W = (filter_size, filter_size, num_in_channel, num_filters)
        # padding = ['SAME', 'VALID'] 'SAME' will keep the same image size with original image size
        layer = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        layer += b
        # don't use sigmoid here
        return tf.nn.relu(layer)


def add_pooling_layer(x, ksize, stride, name):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME',
                          name=name)


def add_flatten_layer(layer):
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


