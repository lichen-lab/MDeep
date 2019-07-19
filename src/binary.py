import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import HAC
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import model
import argparse
tf.reset_default_graph()
tf.set_random_seed(1234)
np.random.seed(1234)


def train (x_train,y_train,args):

    n_classes = 2
    num_epochs = args.max_epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    n_features = x_train.shape[1]
    dropout_rate = args.dropout_rate

    x = tf.placeholder(tf.float32, [None, n_features * 1])
    y = tf.placeholder(tf.float32, [None, n_classes * 1])
    keep_prob = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    x_input = tf.reshape(x, [-1, n_features, 1])


    layer = model.network_binary(x_input,keep_prob,args)


    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer, labels=y))

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.name_scope("accuracy"):

        correct_prediction = tf.equal(tf.argmax(layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    with tf.Session() as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver = tf.train.Saver()
        total_batch = int(np.shape(x_train)[0] / batch_size)

        for epoch in range(num_epochs):

            x_tmp, y_tmp = shuffle(x_train, y_train)
            for i in range(total_batch - 1):
                x_batch, y_true_batch = x_tmp[i * batch_size:i * batch_size + batch_size], \
                                        y_tmp[i * batch_size:i * batch_size + batch_size]

                feed_dict_train = {x: x_batch, y: y_true_batch, keep_prob: dropout_rate}

                sess.run(optimizer, feed_dict=feed_dict_train)

            loos = sess.run(cost, feed_dict={x: x_train, y: y_train, keep_prob: 1})
            train_accuracy = sess.run(accuracy, feed_dict={x: x_train, y: y_train, keep_prob: 1})

            print("Epoch {}, Loss: {:.4f}  Training accuracy:{:.4f}".format(epoch, loos, train_accuracy))

            if train_accuracy > 0.99:
                break

        save_path = saver.save(sess, "./{}/model_binary.ckpt".format(args.model_dir))
        print("Model saved in path: %s" % save_path)


def eval (x_test, y_test, args):

    n_classes = 2
    n_features = x_test.shape[1]

    x = tf.placeholder(tf.float32, [None, n_features * 1])
    y = tf.placeholder(tf.float32, [None, n_classes * 1])
    keep_prob = tf.placeholder(tf.float32)

    x_input = tf.reshape(x, [-1, n_features, 1])

    layer = model.network_binary(x_input,keep_prob,args)


    with tf.name_scope("accuracy"):

        correct_prediction = tf.equal(tf.argmax(layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    with tf.Session() as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver = tf.train.Saver()
        saver.restore(sess, "./{}/model_binary.ckpt".format(args.model_dir))
        outputs = sess.run(layer, feed_dict={x: x_test, y: y_test, keep_prob: 1})
        print(outputs.shape)
        plt.clf()

        fpr, tpr, threshold = metrics.roc_curve(y_test[:,1],outputs[:,1])
        auc = metrics.roc_auc_score(y_test[:,1],outputs[:,1])
        print(auc)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic (AUC ={:.02f})'.format(auc))
        plt.tight_layout()
        plt.savefig(args.result_dir + "/result.jpg")
        plt.show()




def test (x_test, args):

    n_classes = 2
    n_features = x_test.shape[1]

    x = tf.placeholder(tf.float32, [None, n_features * 1])
    y = tf.placeholder(tf.float32, [None, n_classes * 1])
    keep_prob = tf.placeholder(tf.float32)

    x_input = tf.reshape(x, [-1, n_features, 1])

    layer = model.network_binary(x_input,keep_prob,args)

    with tf.Session() as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver = tf.train.Saver()
        saver.restore(sess, "./{}/model_binary.ckpt".format(args.model_dir))
        outputs = sess.run(layer, feed_dict={x: x_test,keep_prob: 1})
        np.savetxt(args.data_dir + '/y_prediction.txt', outputs)

