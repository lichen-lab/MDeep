from __future__ import print_function
import tensorflow as tf



def new_conv1d_layer(input,filters,kernel_size,strides,keep_prob,name, activation, padding):
    with tf.variable_scope(name) as scope:

        layer = tf.layers.conv1d(input, filters, kernel_size, strides, padding=padding)
        if activation=='tanh':
            layer = tf.nn.tanh(layer)
        if activation == 'relu':
            layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, keep_prob=keep_prob, noise_shape=[tf.shape(layer)[0],1,tf.shape(layer)[2]])

        return layer

def new_activation_layer(input, name):
    with tf.variable_scope(name) as scope:

        layer = tf.nn.tanh(input)

        return layer

def new_fc_layer(input, num_inputs, num_outputs, keep_prob, name):
    with tf.variable_scope(name) as scope:
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[num_outputs]))
        dropbout = tf.nn.dropout(input, keep_prob=keep_prob)
        layer = tf.matmul(dropbout, weights) + biases

        return layer


def network_continous(x_input, keep_prob, args):
    num_filter = args.kernel_size
    window_size = args.window_size
    stride_size =  args.strides

    layer = new_conv1d_layer(input=x_input, filters=num_filter[0], kernel_size=window_size[0], strides=stride_size[0], keep_prob=keep_prob,name ="conv1",activation='tanh', padding='valid')

    layer = new_conv1d_layer(input=layer, filters=num_filter[1], kernel_size=window_size[1], strides=stride_size[1], keep_prob=keep_prob,name ="conv2",activation='tanh',padding='valid')

    layer = new_conv1d_layer(input=layer, filters=num_filter[2], kernel_size=window_size[2], strides=stride_size[2], keep_prob=keep_prob,name ="conv3",activation='tanh',padding='valid')

    num_features = layer.get_shape()[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])


    layer = new_fc_layer(input=layer, num_inputs=num_features,keep_prob=keep_prob, num_outputs=64, name="fc1")

    layer = new_activation_layer(layer, name="tanh1")

    layer = new_fc_layer(input=layer, num_inputs=64, keep_prob=keep_prob, num_outputs=8, name="fc2")

    layer = new_activation_layer(layer, name="tanh2")
    layer = new_fc_layer(input=layer, num_inputs=8, keep_prob=keep_prob, num_outputs=1, name="fc3")

    return layer


def network_binary(x_input, keep_prob, args):
    num_filter = args.kernel_size
    window_size = args.window_size
    stride_size =  args.strides
    layer = new_conv1d_layer(input=x_input, filters=num_filter[0], kernel_size=window_size[0], strides=stride_size[0], keep_prob=keep_prob, name="conv1", activation='relu',padding='same')
    layer = tf.layers.batch_normalization(layer, name='bn1')
    layer = new_conv1d_layer(input=layer, filters=num_filter[1], kernel_size=window_size[1], strides=stride_size[1], keep_prob=keep_prob, name="conv2", activation='relu',padding='same')
    layer = tf.layers.batch_normalization(layer, name='bn2')
    num_features = layer.get_shape()[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    layer = new_fc_layer(input=layer, num_inputs=num_features, keep_prob=keep_prob, num_outputs=64, name="fc1")
    layer = tf.layers.batch_normalization(layer, name='bn3')
    layer = new_activation_layer(layer, name="tanh1")
    layer = new_fc_layer(input=layer, num_inputs=64, keep_prob=keep_prob, num_outputs=2, name="fc3")

    return layer
