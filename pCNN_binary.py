from __future__ import print_function
import tensorflow as tf
from sklearn.utils import shuffle
import HAC
import numpy as np
import sys

tf.reset_default_graph()
tf.set_random_seed(1234)
np.random.seed(1234)

n_classes = 2
n_features = 2291
learning_rate =1e-4
num_epochs = 2000
batch_size = 32

x = tf.placeholder(tf.float32, [None, n_features*1])
y = tf.placeholder(tf.float32, [None, n_classes*1])
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

x_input = tf.reshape(x, [-1, n_features, 1])

def new_conv1d_layer(input,filters,kernel_size,strides,keep_prob,name):
    with tf.variable_scope(name) as scope:

        layer = tf.layers.conv1d(input, filters, kernel_size, strides, padding='same')
        layer = tf.nn.dropout(layer, keep_prob=keep_prob, noise_shape=[tf.shape(layer)[0],1,tf.shape(layer)[2]])
        layer = tf.nn.relu(layer)
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


layer = new_conv1d_layer(input=x_input, filters=32, kernel_size=128, strides=64, keep_prob=keep_prob,name ="conv1")
layer = tf.layers.batch_normalization(layer,name='bn1')
layer = new_conv1d_layer(input=layer, filters=32, kernel_size=4, strides=2, keep_prob=keep_prob,name ="conv2",)
layer = tf.layers.batch_normalization(layer,name='bn2')
num_features = layer.get_shape()[1:4].num_elements()
layer = tf.reshape(layer, [-1, num_features])

layer = new_fc_layer(input=layer, num_inputs=num_features,keep_prob=keep_prob, num_outputs=64, name="fc1")
layer = tf.layers.batch_normalization(layer,name='bn3')
layer = new_activation_layer(layer, name="tanh1")
layer = new_fc_layer(input=layer, num_inputs=64, keep_prob=keep_prob, num_outputs=2, name="fc3")


l2_regularizer = tf.contrib.layers.l2_regularizer(
   scale=0.001, scope=None
)
l1_regularizer = tf.contrib.layers.l1_regularizer(
   scale=0.001, scope=None
)


weights = tf.trainable_variables()
l1_regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
with tf.name_scope("loss"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer, labels=y))
    # cost = cost +l2_regularization_penalty
    # cost =  cost + l1_regularization_penalty
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.name_scope("accuracy"):

    print_y = y
    y_score = tf.nn.softmax(logits=layer)
    print_layer = y_score
    correct_prediction = tf.equal(tf.argmax(layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y,1),
                                      predictions=tf.argmax(layer,1))
    auc, auc_op = tf.metrics.auc(labels=y[:,1], predictions=layer[:,1])


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


            loos = sess.run(cost, feed_dict={x: x_train, y: y_train, keep_prob: 1})
            train_accuracy = sess.run(accuracy, feed_dict={x: x_train, y: y_train, keep_prob: 1})

            print("Epoch {}, Loss: {:.4f}  Training accuracy:{:.4f}".format(epoch, loos, train_accuracy))

            if train_accuracy > 0.99:
                test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1})
                print("Test accuracy:{}".format(test_accuracy))
                break

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
    y_tr = Y[:cut]
    x_test = X[cut:, :]
    y_te = Y[cut:]

    x_train = x_train[:, hac_index]
    x_test = x_test[:, hac_index]

    y_train = []
    for l in y_tr:
        if l == 1:
            y_train.append([0, 1])
        else:
            y_train.append([1, 0])
    y_train = np.array(y_train, dtype=int)

    y_test = []
    for l in y_te:
        if l == 1:
            y_test.append([0, 1])
        else:
            y_test.append([1, 0])

    y_test = np.array(y_test, dtype=int)

    train(x_train,y_train, x_test, y_test)
