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
    parser.add_argument('-lr', type=numpy.float32, default=0.1)
    parser.add_argument('-filter_shape', type=Shape, nargs='+', default=[(64, 3, 3), (2, 2)])
    parser.add_argument('-kernel_size', type=int, default=3)
    parser.add_argument('-kernel_pool_size', type=int, default=-1)
    parser.add_argument('-batch_size', type=int, default=200)
    parser.add_argument('-load_model', type=int, default=0)
    parser.add_argument('-save_model', type=str, default='model.ckpt')
    parser.add_argument('-train_on_valid', type=int, default=1)
    parser.add_argument('-conv_type', type=str, default='standard') # standard
    parser.add_argument('-learning_decay', type=numpy.float32, default=0.5)
    parser.add_argument('-keep_prob', type=str, default=0.5)
    parser.add_argument('-save_dir', type=str, default='../save')
    parser.add_argument('-path_log', type=str, default='../logs')
    parser.add_argument('-use_log', type=int, default=1)

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

def restore_model(saver, session, loadfrom):
    saver.restore(session, loadfrom)
    print "Model restored from " + loadfrom

def store_model(saver, session, saveto):
    save_path = saver.save(session, saveto)
    print "Model saved to " + saveto



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
    use_log = opt['use_log']

    # Save or load the model
    if save_model is not 'none':
        saveto = '../save/' + dataset + '_' + save_model
    else:
        saveto = None
    
    if load_model:
        loadfrom = saveto # load from the file specified by the <save_model> argument and will save to the same file
    else:
        loadfrom = None

    [(train_x, train_y, train_num),
     (valid_x, valid_y, valid_num),
     (test_x, test_y, test_num),
     num_class, image_shape] = load_normalize_data(dataset)

    model = Model(image_shape, filter_shape, num_class, conv_type, kernel_size, kernel_pool_size)


    n_train_batches = train_num/ batch_size
    #n_train_batches = 20/batch_size
    n_valid_batches = valid_num/ batch_size
    n_test_batches = test_num/ batch_size

    print "Training..."

    train_errors = []
    valid_errors = []
    train_costs = []
    valid_costs = []
    test_errors = []

    best_valid_err = 1.
    best_valid_epoch = 0
    bad_count = 0


    with tf.Session() as sess:
        if use_log:
            train_writer = tf.summary.FileWriter(path_log + '/train', sess.graph)
            valid_writer = tf.summary.FileWriter(path_log + '/valid')
            # tensorboard --logdir=path/to/logs

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        # Load the variables of the model if wanted
        # Do we have to initialize the variables before loading them?
        if loadfrom:
            print "Loading model..."
            restore_model(saver, sess, loadfrom)
            

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
                # batch_idx = idx_perm[batch*batch_size:(batch+1)*batch_size]
                # train_batch_x = train_x[batch_idx]
                # train_batch_y = train_y[batch_idx]
                train_batch_x = train_x[batch*batch_size:(batch+1)*batch_size]
                train_batch_y = train_y[batch*batch_size:(batch+1)*batch_size]

                feed = {model.inputs: train_batch_x,
                        model.targets: train_batch_y,
                        model.keep_prob: keep_prob}
                _ = sess.run([model.train_op], feed)
            
            for batch in range(n_train_batches):
                train_batch_x = train_x[batch*batch_size:(batch+1)*batch_size]
                train_batch_y = train_y[batch*batch_size:(batch+1)*batch_size]

                feed = {model.inputs: train_batch_x,
                        model.targets: train_batch_y,
                        model.keep_prob: 1.}
                train_loss, train_err = sess.run([model.loss, model.err], feed)
                cur_costs.append(train_loss)
                cur_errors.append(train_err)

            loss = numpy.mean(cur_costs)
            err = numpy.mean(cur_errors)

            # used in tensorboard
            if use_log:
                summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss)])
                train_writer.add_summary(summary, epoch)
                summary = tf.Summary(value=[tf.Summary.Value(tag="error", simple_value=err)])
                train_writer.add_summary(summary, epoch)

            train_errors.append(err)
            train_costs.append(loss)

            # compute validation loss and err
            cur_costs = []
            cur_errors = []
            for batch in range(n_valid_batches):
                valid_batch_x = valid_x[batch*batch_size:(batch+1)*batch_size]
                valid_batch_y = valid_y[batch*batch_size:(batch+1)*batch_size]
                feed = {model.inputs: valid_batch_x,
                        model.targets: valid_batch_y,
                        model.keep_prob: 1.}
                valid_loss, valid_err = sess.run([model.loss, model.err], feed)
                cur_costs.append(valid_loss)
                cur_errors.append(valid_err)

            loss = numpy.mean(cur_costs)
            err = numpy.mean(cur_errors)

            # used in tensorboard
            if use_log:
                summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss)])
                valid_writer.add_summary(summary, epoch)
                summary = tf.Summary(value=[tf.Summary.Value(tag="error", simple_value=err)])
                valid_writer.add_summary(summary, epoch)


            valid_errors.append(err)
            valid_costs.append(loss)


            # compute test loss and err for best model
            # take 22s too long
            cur_errors = []
            for batch in range(n_test_batches):
                test_batch_x = test_x[batch * batch_size:(batch + 1) * batch_size]
                test_batch_y = test_y[batch * batch_size:(batch + 1) * batch_size]
                feed = {model.inputs: test_batch_x,
                        model.targets: test_batch_y,
                        model.keep_prob: 1.}
                test_err = sess.run([model.err], feed)
                cur_errors.append(test_err)
            #
            # # todo add to tensorboard
            # test_errors.append(numpy.mean(cur_errors))

            # keep best model and reduce learning rate if necessary
            if valid_errors[-1] <= best_valid_err:
                best_valid_err = valid_errors[-1]
                best_valid_epoch = epoch
                bad_count = 0
                # todo copy current model as current best model
                best_sess = sess # Useless for now but to check if mutable or not
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
            if saveto and (epoch%10 == 0) and (epoch>0):
                print 'Saving model ...'
                store_model(saver, sess, saveto)       

        # Save last model
        store_model(saver, sess, saveto)

        print 'Best errors train {:.4f}, valid {:.4f}, test {:.4f}'.format(train_errors[best_valid_epoch], valid_errors[best_valid_epoch], test_errors[best_valid_epoch])


        # todo save err, cost
        # todo add visialisation of the weights

        print ">> tensorboard --logdir={}".format(path_log)
        if use_log:
            train_writer.close()
            valid_writer.close()

if __name__ == '__main__':
    args = parseArgs()
    train(args)












