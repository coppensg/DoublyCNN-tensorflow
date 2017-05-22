# file used for training
# code mainly inspired from https://github.com/Shuangfei/doublecnn/blob/master/main.py
import sys
import os
import time
from six.moves import cPickle
import argparse
import numpy
import tensorflow as tf

import utils
from model import Model


def Shape(s):
    return tuple(map(int, s.split(',')))


def parseArgs():
    parser = argparse.ArgumentParser()

    # Code params
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('-s', '--save_dir', type=str, default='./save')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('-m', '--model_file', type=str, default=None)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('-l', '--path_log', type=str, default='./logs')

    # Model params
    parser.add_argument('-t', '--conv_type', type=str, default='standard')  # standard or double
    parser.add_argument('-filter_shape', type=Shape, nargs='+', default=[(10, 3, 3)])
    parser.add_argument('-kernel_size', type=int, default=3)
    parser.add_argument('-kernel_pool_size', type=int, default=-1)

    # Training params
    parser.add_argument('-b', '--batch_size', type=int, default=20)
    parser.add_argument('-e', '--train_epochs', type=int, default=150)
    parser.add_argument('-p', '--patience', type=int, default=10)
    parser.add_argument('-lr', type=numpy.float32, default=0.1)
    parser.add_argument('-learning_decay', type=numpy.float32, default=0.5)
    parser.add_argument('-keep_prob', type=str, default=0.5)
    parser.add_argument('-augmentation', action='store_true')

    args = parser.parse_args()
    # Save config
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)

    return vars(args)


def training(sess, model, opt, train, valid, save):
    # Model data
    batch_size = opt['batch_size']
    train_epochs = opt['train_epochs']
    patience = opt['patience']
    lr = opt['lr']
    learning_decay = opt['learning_decay']
    keep_prob = opt['keep_prob']
    use_log = opt['log']
    path_log = opt['path_log']
    augmentation = opt['augmentation']

    if augmentation:
        print 'Applying data augmentation at each epoch'
    else:
        print 'No data augmentation'

    (train_x, train_y, train_num) = train
    (valid_x, valid_y, valid_num) = valid

    n_train_batches = train_num / batch_size
    n_valid_batches = valid_num / batch_size

    tr_appender = utils.factory_appender(sess=sess, use_log=use_log, log_dir=path_log, log_filename="train")
    va_appender = utils.factory_appender(sess=sess, use_log=use_log, log_dir=path_log, log_filename="valid")

    train_errors, valid_errors = numpy.zeros((train_epochs)), numpy.zeros((train_epochs))
    train_costs, valid_costs = numpy.zeros((train_epochs)), numpy.zeros((train_epochs))
    learning_rates = numpy.zeros((train_epochs))

    best_valid_err = 1.
    best_valid_epoch = 0
    bad_count = 0

    print "Training..."
    sess.run(tf.assign(model.lr, lr))
    for epoch in range(train_epochs):
        start = time.time()

        # shuffle the train set
        idx_perm = numpy.random.permutation(train_num)
        train_x = train_x[idx_perm]
        train_y = train_y[idx_perm]

        if augmentation:
            beg = time.time()
            tensor_x = utils.distord(train_x)
            train_x = numpy.array(sess.run([tensor_x]))[0]
            end = time.time()
            print 'data_augmentation/epoch = {:.3f} s'.format(end - beg)

        # compute train loss, err and update weights
        utils.update_model(sess=sess, model=model, inputs=train_x, target=train_y,
                           batch_size=batch_size, n_batch=n_train_batches, keep_prob=keep_prob)

        # Compute training loss and err
        loss, err = utils.fwd_eval(sess=sess, model=model, inputs=train_x, target=train_y,
                                   batch_size=batch_size, n_batch=n_train_batches)
        tr_appender(train_errors, err, epoch, "error")
        tr_appender(train_costs, loss, epoch, "loss")

        # compute validation loss and err
        loss, err = utils.fwd_eval(sess=sess, model=model, inputs=valid_x, target=valid_y,
                                   batch_size=batch_size, n_batch=n_valid_batches)
        va_appender(valid_errors, err, epoch, "error")
        va_appender(valid_costs, loss, epoch, "loss")

        current_lr = sess.run(model.lr)
        tr_appender(learning_rates, current_lr, epoch, "learning_rate")

        # keep best model and reduce learning rate if necessary
        if valid_errors[epoch] <= best_valid_err:
            best_valid_err = valid_errors[epoch]
            best_valid_epoch = epoch
            # update the current best model
            save()
        else:
            print "bad_count"
            bad_count += 1
            if bad_count > patience:
                print 'Reducing the learning rate..'
                sess.run(tf.assign(model.lr, model.lr * learning_decay))
                bad_count = 0

        end = time.time()
        print "epoch. {}, train_loss = {:.4f}, valid_loss = {:.4f}," \
              "train_error = {:.4f}, valid_error = {:.4f}, time/epoch = {:.3f} s" \
            .format(epoch, train_costs[epoch], valid_costs[epoch], train_errors[epoch],
                    valid_errors[epoch], end - start)

    print 'Best errors train {:.4f}, valid {:.4f}'.format(train_errors[best_valid_epoch],
                                                          valid_errors[best_valid_epoch])


def main(arg):
    # Program Parameters
    dataset = arg['dataset']
    save_dir = arg['save_dir']
    load_model = arg['load']
    model_filename = arg['model_file']

    # Topology model
    conv_type = arg['conv_type']
    filter_shape = arg['filter_shape']
    kernel_size = arg['kernel_size']
    kernel_pool_size = arg['kernel_pool_size']

    # Data
    [train, valid, _, num_class, image_shape] = utils.load_normalize_data(dataset)

    # Save/load
    saveto = os.path.join(save_dir, model_filename) if model_filename is not None else None
    loadfrom = saveto if load_model else None

    # Topology
    model = Model(image_shape, filter_shape, num_class, conv_type, kernel_size, kernel_pool_size)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        save = utils.store_model(saver, sess, saveto)

        # Load the variables of the model if wanted
        if load_model:
            print "Loading model..."
            utils.restore_model(saver, sess, loadfrom)
        else:
            sess.run(tf.global_variables_initializer())

        training(sess, model, arg, train, valid, save)


if __name__ == '__main__':
    args = parseArgs()

    if args["log"]:
        print ">> tensorboard --logdir={}".format(args['path_log'])

    main(args)
