from __future__ import print_function
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import HAC
import sys

tf.reset_default_graph()
tf.set_random_seed(1234)
np.random.seed(1234)

learning_rate =5e-3
num_epochs = 2000
batch_size = 16
n_classes = 1
n_features = 1087

x = tf.placeholder(tf.float32, [None, n_features*1])
y = tf.placeholder(tf.float32, [None, n_classes*1])
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

x_input = tf.reshape(x, [-1, n_features, 1])


def new_conv1d_layer(input,filters,kernel_size,strides,keep_prob,name):
    with tf.variable_scope(name) as scope:

        layer = tf.layers.conv1d(input, filters, kernel_size, strides, padding='valid')
        layer = tf.nn.tanh(layer)

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

layer = new_conv1d_layer(input=x_input, filters=64, kernel_size=8, strides=4, keep_prob=keep_prob,name ="conv1")

layer = new_conv1d_layer(input=layer, filters=64, kernel_size=8, strides=4, keep_prob=keep_prob,name ="conv2")

layer = new_conv1d_layer(input=layer, filters=32, kernel_size=8, strides=4, keep_prob=keep_prob,name ="conv3")

num_features = layer.get_shape()[1:4].num_elements()
layer = tf.reshape(layer, [-1, num_features])


layer = new_fc_layer(input=layer, num_inputs=num_features,keep_prob=keep_prob, num_outputs=64, name="fc1")

layer = new_activation_layer(layer, name="tanh1")

layer = new_fc_layer(input=layer, num_inputs=64, keep_prob=keep_prob, num_outputs=8, name="fc2")

layer = new_activation_layer(layer, name="tanh2")
layer = new_fc_layer(input=layer, num_inputs=8, keep_prob=keep_prob, num_outputs=1, name="fc3")


l2_regularizer = tf.contrib.layers.l2_regularizer(
   scale=0.05, scope=None
)
l1_regularizer = tf.contrib.layers.l1_regularizer(
   scale=0.0001, scope=None
)

weights = tf.trainable_variables()
l1_regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)

with tf.name_scope("mse"):
    cost = tf.reduce_mean(tf.square(y - layer))
    cost = cost+l2_regularization_penalty

with tf.name_scope("optimizer"):

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope("accuracy"):
    mse = tf.reduce_mean(tf.square(y - layer))
    output = layer
    cor = tf.contrib.metrics.streaming_pearson_correlation(layer,y)



def train(x_train,y_train, x_test, y_test):
    with tf.Session() as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        total_batch = int(np.shape(x_train)[0] / batch_size)

        for epoch in range(num_epochs):

            x_tmp, y_tmp = shuffle(x_train, y_train)
            for i in range(total_batch-1):

                x_batch, y_true_batch = x_tmp[i*batch_size:i*batch_size+batch_size], \
                                    y_tmp[i*batch_size:i*batch_size+batch_size]
                feed_dict_train = {x: x_batch, y: y_true_batch, keep_prob:0.5}

                sess.run(optimizer, feed_dict=feed_dict_train)


            y_train_hat = sess.run(output, feed_dict={x: x_train, y: y_train, keep_prob: 1})
            cor_train = np.corrcoef(y_train_hat.reshape(y_train.shape[0],), y_train.reshape(y_train.shape[0],))

            train_mse = sess.run(cost, feed_dict={x:x_train, y:y_train, keep_prob:1})

            print("Epoch {}: Training loss:{}, Training cor:{}".format(epoch,train_mse,cor_train[0,1]))

            if cor_train[0,1] > 0.99:
                break

        outputs = sess.run(output, feed_dict={x: x_test, y: y_test, keep_prob: 1})
        cor_test = np.corrcoef(outputs.reshape(y_test.shape[0], ), y_test.reshape(y_test.shape[0], ))
        print("Test cor:{}".format(cor_test[0, 1]))


if __name__ == '__main__':

    file = './{}/'.format(sys.argv[1])

    C = np.load(file + 'c.npy')
    X = np.load(file + 'X.npy')
    Y = np.load(file + 'Y.npy')
    print("Hierarchical clustering")
    hac_index = HAC.hac(C)
    print("Start training")

    cut = int(X.shape[0] * 0.8)
    x_train = X[:cut, :]
    y_train = Y[:cut]
    x_test = X[cut:, :]
    y_test = Y[cut:]

    x_train = x_train[:, hac_index]
    x_test = x_test[:, hac_index]

    train(x_train, y_train, x_test, y_test)
