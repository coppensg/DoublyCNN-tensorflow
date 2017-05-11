# file used for training
# code mainly inspired from https://github.com/Shuangfei/doublecnn/blob/master/main.py
import sys
import os
import time
from six.moves import cPickle
import argparse
import numpy
import tensorflow as tf

import load_data_cifar10
import load_data_cifar100

from model import Model


def Shape(s):
    return tuple(map(int, s.split(',')))

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cifar10')
    parser.add_argument('-train_epochs', type=int, default=150)
    parser.add_argument('-patience', type=int, default=10)
    parser.add_argument('-lr', type=numpy.float32, default=1e0)
    parser.add_argument('-filter_shape', type=Shape, nargs='+', default=[(64, 7, 7), (2, 2)])
    parser.add_argument('-kernel_size', type=int, default=5)
    parser.add_argument('-kernel_pool_size', type=int, default=-1)
    parser.add_argument('-batch_size', type=int, default=200)
    parser.add_argument('-load_model', type=int, default=0)
    parser.add_argument('-save_model', type=str, default='model_saved.npz')
    parser.add_argument('-train_on_valid', type=int, default=1)
    parser.add_argument('-conv_type', type=str, default='double') # standard
    parser.add_argument('-learning_decay', type=numpy.float32, default=0.5)
    parser.add_argument('-keep_prob', type=str, default=0.5)
    parser.add_argument('-save_dir', type=str, default='../save')
    parser.add_argument('-path_log', type=str, default='../logs')

    args = parser.parse_args()
    # Save config
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)

    return args


def load_normalize_data(dataset):

    if dataset == 'cifar10':
        data_dir = load_data_cifar10.data_dir
        load_data = load_data_cifar10.load_data
    elif dataset == 'cifar100':
        load_data = load_data_cifar100.load_data
        data_dir = load_data_cifar100.data_dir
    else:
        sys.exit("The dataset {} is not recognized".format(dataset))

    image_shape = (3, 32, 32)
    print "Load the dataset {}...".format(dataset)
    datasets = load_data(data_dir)
    train_x, valid_x, test_x = [data[0] for data in datasets]
    train_y, valid_y, test_y = [data[1] for data in datasets]

    # one hot encoding
    num_class = numpy.max(train_y) + 1
    train_y = numpy.eye(num_class, dtype=float)[train_y]
    valid_y = numpy.eye(num_class, dtype=float)[valid_y]
    test_y = numpy.eye(num_class, dtype=float)[test_y]

    train_num, valid_num, test_num = [data[0].shape[0] for data in datasets]

    print "Normalize the training, validation, test sets..."
    xmean = train_x.mean(axis=0)
    train_x -= xmean
    valid_x -= xmean
    test_x -= xmean

    return [(train_x, train_y, train_num), (valid_x, valid_y, valid_num), (test_x, test_y, test_num), num_class, image_shape]

def train(args):

    opt = vars(args)
    dataset = opt['dataset']
    train_epochs = opt['train_epochs']
    patience = opt['patience']
    lr = opt['lr']
    filter_shape = opt['filter_shape']
    kernel_size = opt['kernel_size']
    kernel_pool_size = opt['kernel_pool_size']
    batch_size = opt['batch_size']
    load_model = opt['load_model']
    save_model = opt['save_model']
    conv_type = opt['conv_type']
    keep_prob = opt['keep_prob']
    learning_decay = opt['learning_decay']
    path_log = opt['path_log']

    # todo to add
    # if save_model is not 'none':
    #     saveto = './saved/' + dataset + '_' + save_model
    # else:
    #     saveto = None
    #
    # if load_model:
    #     loadfrom = saveto
    # else:
    #     loadfrom = None

    [(train_x, train_y, train_num),
     (valid_x, valid_y, valid_num),
     (test_x, test_y, test_num),
     num_class, image_shape] = load_normalize_data(dataset)

    model = Model(image_shape, filter_shape, num_class, conv_type, kernel_size, kernel_pool_size)

    # todo add the option to load and save a model

    n_train_batches = 200/batch_size#train_num/ batch_size
    n_valid_batches = valid_num/ batch_size
    n_test_batches = test_num/ batch_size

    print "Training..."

    train_errors = []
    valid_errors = []
    train_costs = []
    valid_costs = []

    best_valid_err = 1.
    best_valid_epoch = 0
    bad_count = 0
    best_model = model

    with tf.Session() as sess:
        file_writer = tf.summary.FileWriter(path_log, sess.graph)
        # tensorboard --logdir=path/to/logs
        sess.run(tf.global_variables_initializer())


        # todo load/ restore model

        sess.run(tf.assign(model.lr, lr))
        for epoch in range(train_epochs):
            start = time.time()
            # shuffle the train set
            idx_perm = numpy.random.permutation(train_num)

            # compute train loss, err and update weights
            cur_costs = []
            cur_errors = []
            for batch in range(n_train_batches):
                # create batch
                batch_idx = idx_perm[batch*batch_size:(batch+1)*batch_size]
                train_batch_x = train_x[batch_idx]
                train_batch_y = train_y[batch_idx]

                feed = {model.inputs: train_batch_x,
                        model.targets: train_batch_y,
                        model.keep_prob: keep_prob}
                _ = sess.run([model.train_op], feed)

                feed = {model.inputs: train_batch_x,
                        model.targets: train_batch_y,
                        model.keep_prob: 1.}
                train_loss, train_err = sess.run([model.loss, model.err], feed)

                cur_costs.append(train_loss)
                cur_errors.append(train_err)

            # todo add to tensorboard
            train_errors.append(numpy.mean(cur_errors))
            train_costs.append(numpy.mean(cur_costs))

            # compute validation loss and err
            cur_costs = []
            cur_errors = []
            for batch in range(n_valid_batches):
                valid_batch_x = train_x[batch*batch_size:(batch+1)*batch_size]
                valid_batch_y = train_y[batch*batch_size:(batch+1)*batch_size]
                feed = {model.inputs: valid_batch_x,
                        model.targets: valid_batch_y,
                        model.keep_prob: 1.}
                valid_loss, valid_err = sess.run([model.loss, model.err], feed)
                cur_costs.append(valid_loss)
                cur_errors.append(valid_err)

            # todo add to tensorboard
            valid_errors.append(numpy.mean(cur_errors))
            valid_costs.append(numpy.mean(cur_costs))


            # keep best model and reduce learning rate if necessary
            if valid_errors[-1] <= best_valid_err:
                best_valid_err = valid_errors[-1]
                best_valid_epoch = epoch
                bad_count = 0
                # todo copy current model as current best model
            else:
                bad_count += 1
                if bad_count > patience:
                    print 'Reducing the learning rate..'
                    lr = lr*learning_decay
                    sess.run(tf.assign(model.lr, lr))
                    bad_count = 0
            end = time.time()
            print "epoch. {}, train_loss = {:.4f}, valid_loss = {:.4f}, train_error = {:.4f}, valid_error = {:.4f}, time/epoch = {:.3f} s" \
                .format(epoch, train_costs[-1], valid_costs[-1], train_errors[-1], valid_errors[-1], end - start)

        # save models

        # compute test loss and err for best model
        cur_costs = []
        cur_errors = []
        for batch in range(n_test_batches):
            test_batch_x = test_x[batch * batch_size:(batch + 1) * batch_size]
            test_batch_y = test_y[batch * batch_size:(batch + 1) * batch_size]
            feed = {model.inputs: test_batch_x,
                    model.targets: test_batch_y,
                    model.keep_prob: 1.}
            test_err = sess.run([model.err], feed)


        print 'Best errors train {:.4f}, valid {:.4f}, test {:.4f}'.format(train_errors[best_valid_epoch], valid_errors[best_valid_epoch], test_err)


        # todo save err, cost
        # todo add visialisation of the weights

        print ">> tensorboard --logdir={}".format(path_log)
        file_writer.close()

if __name__ == '__main__':
    args = parseArgs()
    train(args)












