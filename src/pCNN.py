from __future__ import print_function
import numpy as np
import HAC
import argparse



def main(args):

    file = args.data_dir
    training = args.train
    C = np.load('./' + file + '/c.npy')
    x_train = np.load('./' +file+'/X_train.npy')

    y_train = np.load('./' +file+'/Y_train.npy')
    x_test = np.load('./' +file+'/X_test.npy')
    y_test = np.load('./' +file+'/Y_test.npy')

    print("Hierarchical clustering")
    hac_index = HAC.hac(C)

    x_train = x_train[:, hac_index]
    x_test = x_test[:, hac_index]

    if args.outcome_type == 'continous':
        import continous
        if training:
            print("Start training")
            continous.train(x_train, y_train, args)
        else:
            print("Start testing")
            continous.test(x_test,y_test, args)

    elif args.outcome_type == 'binary':
        import binary
        y_tr = []
        for l in y_train:
            if l == 1:
                y_tr.append([0, 1])
            else:
                y_tr.append([1, 0])
        y_tr = np.array(y_tr, dtype=int)

        y_te = []
        for l in y_test:
            if l == 1:
                y_te.append([0, 1])
            else:
                y_te.append([1, 0])

        y_te = np.array(y_te, dtype=int)

        if training:
            print("Start training")
            binary.train(x_train, y_tr, args)
        else:
            print("Start testing")
            binary.test(x_test,y_te, args)



def parse_arguments(parser):
    parser.add_argument('--data_dir', type=str, default='data/USA', metavar='<data_directory>',
                        help='The data directory')

    parser.add_argument('--model_dir', type=str, default='model',
                        help='The directory to save or restore the trained models.')

    parser.add_argument('--train', dest='train', action='store_true', help='Use this option for training model')
    parser.add_argument('--test', dest='train', action='store_false',help='Use this option for testing model')
    parser.set_defaults(train=True)

    parser.add_argument('--outcome_type', type=str, default='continous',
                        help='The outcome type')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size for training')

    parser.add_argument('--max_epoch', type=int, default=2000,
                        help='The max epoch for training')

    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='The learning rate for training')

    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='The dropout rate for training')

    parser.add_argument('--L2_regularizer', type=float, default=0.05,
                        help='The L2 lambda')


    parser.add_argument('--window_size', nargs='+' ,type=int, default=[8,8,8],
                        help='The window size for convolutional layers')

    parser.add_argument('--kernel_size', nargs='+' ,type=int, default=[64, 64, 32],
                        help='The kernel size for convolutional layers')

    parser.add_argument('--strides',nargs='+' ,type=int, default=[4, 4, 4],
                        help='The strides size for convolutional layers')



    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A Phylogeny-regularized Convolutional NeuralNetwork for Microbiome-based Predictions')

    args = parse_arguments(parser)
    print(args)
    main(args)
