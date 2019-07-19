from __future__ import print_function
import numpy as np
import HAC
import argparse



def main(args):

    file = args.data_dir
    training = args.train
    testing = args.test
    evaluation = args.evaluation
    C = np.load('./' + file + '/c.npy')
    print("Hierarchical clustering")
    hac_index = HAC.hac(C)

    if args.outcome_type == 'continous':
        import continous

        if training:
            print("Start training")
            x_train = np.load('./' + file + '/X_train.npy')
            y_train = np.load('./' + file + '/Y_train.npy')
            x_train = x_train[:, hac_index]
            y_train = y_train.reshape(y_train.shape[0],1)

            continous.train(x_train, y_train, args)

        if evaluation:
            print("Start evaluation")
            x_test = np.load('./' + file + '/X_eval.npy')
            y_test = np.load('./' + file + '/Y_eval.npy')
            x_test = x_test[:, hac_index]
            y_test = y_test.reshape(y_test.shape[0],1)
            continous.eval(x_test,y_test, args)

        if testing:
            print("Start testing")
            x_test = np.load(args.test_file)
            x_test = x_test[:, hac_index]
            continous.test(x_test,args)

    elif args.outcome_type == 'binary':
        import binary

        if training:
            print("Start training")
            x_train = np.load('./' + file + '/X_train.npy')
            y_train = np.load('./' + file + '/Y_train.npy')

            y_tr = []
            for l in y_train:
                if l == 1:
                    y_tr.append([0, 1])
                else:
                    y_tr.append([1, 0])
            y_tr = np.array(y_tr, dtype=int)


            x_train = x_train[:, hac_index]
            binary.train(x_train, y_tr, args)

        if evaluation:
            print("Start evaluation")
            x_test = np.load('./' + file + '/X_eval.npy')
            y_test = np.load('./' + file + '/Y_eval.npy')
            x_test = x_test[:, hac_index]
            y_te = []
            for l in y_test:
                if l == 1:
                    y_te.append([0, 1])
                else:
                    y_te.append([1, 0])
            y_te = np.array(y_te, dtype=int)
            binary.eval(x_test,y_te, args)

        if testing:
            print("Start testing")
            x_test = np.load(args.test_file)
            x_test = x_test[:, hac_index]
            binary.test(x_test,args)

def parse_arguments(parser):

    parser.add_argument('--train', dest='train', action='store_true', help='Use this option for train model')
    parser.set_defaults(train=False)

    parser.add_argument('--evaluation', dest='evaluation', action='store_true',help='Use this option for evaluate model')
    parser.set_defaults(evaluation=False)

    parser.add_argument('--test', dest='test', action='store_true', help='Use this option for test model')
    parser.set_defaults(test=False)

    parser.add_argument('--data_dir', type=str, default='data/USA', metavar='<data_directory>',
                        help='The data directory')

    parser.add_argument('--test_file', type=str, default='data/USA/X_test.npy', help='The unlabelled test file')

    parser.add_argument('--model_dir', type=str, default='model',
                        help='The directory to save or restore the trained models.')

    parser.add_argument('--result_dir', type=str, default='result/USA', metavar='<data_directory>',
                        help='The directory to save test / evaluation result')

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
