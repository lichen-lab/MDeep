import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import HAC
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import model
tf.reset_default_graph()
tf.set_random_seed(1234)
np.random.seed(1234)


def train(x_train,y_train,args):
    n_classes = 1
    num_epochs = args.max_epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    l2_lambda = args.L2_regularizer
    n_features = x_train.shape[1]
    dropout_rate = args.dropout_rate


    x = tf.placeholder(tf.float32, [None, n_features * 1])
    y = tf.placeholder(tf.float32, [None, n_classes * 1])
    keep_prob = tf.placeholder(tf.float32)

    x_input = tf.reshape(x, [-1, n_features, 1])

    layer = model.network_continous(x_input, keep_prob, args)

    l2_regularizer = tf.contrib.layers.l2_regularizer(
        scale= l2_lambda, scope=None
    )

    weights = tf.trainable_variables()
    l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)

    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.square(y - layer))
        cost = cost + l2_regularization_penalty

    with tf.name_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


    with tf.Session() as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver = tf.train.Saver()
        total_batch = int(np.shape(x_train)[0] / batch_size)

        for epoch in range(num_epochs):

            x_tmp, y_tmp = shuffle(x_train, y_train)
            for i in range(total_batch-1):

                x_batch, y_true_batch = x_tmp[i*batch_size:i*batch_size+batch_size], \
                                    y_tmp[i*batch_size:i*batch_size+batch_size]
                feed_dict_train = {x: x_batch, y: y_true_batch, keep_prob:dropout_rate}

                sess.run(optimizer, feed_dict=feed_dict_train)


            y_train_hat = sess.run(layer, feed_dict={x: x_train, y: y_train, keep_prob: 1})
            cor_train = np.corrcoef(y_train_hat.reshape(y_train.shape[0],), y_train.reshape(y_train.shape[0],))
            train_mse = sess.run(cost, feed_dict={x:x_train, y:y_train, keep_prob:1})

            print("Epoch {}: Training loss:{}, Training cor:{}".format(epoch,train_mse,cor_train[0,1]))

            if cor_train[0,1] > 0.99:
                break

        save_path = saver.save(sess, "./{}/model_continous.ckpt".format(args.model_dir))
        print("Model saved in path: %s" % save_path)


def eval(x_test, y_test, args):

    n_classes = 1
    n_features = x_test.shape[1]

    x = tf.placeholder(tf.float32, [None, n_features * 1])
    y = tf.placeholder(tf.float32, [None, n_classes * 1])
    keep_prob = tf.placeholder(tf.float32)

    x_input = tf.reshape(x, [-1, n_features, 1])
    layer = model.network_continous(x_input, keep_prob, args)

    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.square(y - layer))

    with tf.Session() as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver = tf.train.Saver()
        saver.restore(sess,  "./{}/model_continous.ckpt".format(args.model_dir))
        test_loss =sess.run(cost, feed_dict={x: x_test, y: y_test, keep_prob: 1})
        outputs = sess.run(layer, feed_dict={x: x_test, y: y_test, keep_prob: 1})
        cor_test = np.corrcoef(outputs.reshape(y_test.shape[0], ), y_test.reshape(y_test.shape[0], ))
        print("Test loss: {}, Test cor:{}".format(test_loss, cor_test[0, 1]))
        plt.clf()
        ax = sns.scatterplot(x =y_test.reshape(y_test.shape[0], ), y = outputs.reshape(y_test.shape[0], ), marker='+')

        x_line = np.linspace(max(min(outputs.reshape(y_test.shape[0], )), min(y_test.reshape(y_test.shape[0], ))),
                             min(max(outputs.reshape(y_test.shape[0], )), max(y_test.reshape(y_test.shape[0], ))))
        plt.plot(x_line, x_line)
        plt.xlabel('Y')
        plt.ylabel(r'Predicted Y')
        plt.title(r'Test result on Pretrained pCNN model ($R^2$ = {:.02f})'.format(cor_test[0, 1] ** 2))
        plt.tight_layout()
        ax.figure.savefig( args.result_dir + "/result.jpg")
        plt.show()


def test(x_test, args):

    n_classes = 1
    n_features = x_test.shape[1]

    x = tf.placeholder(tf.float32, [None, n_features * 1])
    y = tf.placeholder(tf.float32, [None, n_classes * 1])
    keep_prob = tf.placeholder(tf.float32)

    x_input = tf.reshape(x, [-1, n_features, 1])
    layer = model.network_continous(x_input, keep_prob, args)

    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.square(y - layer))

    with tf.Session() as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver = tf.train.Saver()
        saver.restore(sess,  "./{}/model_continous.ckpt".format(args.model_dir))
        outputs = sess.run(layer, feed_dict={x: x_test, keep_prob: 1})
        np.savetxt(args.result_dir + '/y_prediction.txt', outputs)
        print('Prediction.txt is saved in {}'.format(args.result_dir))