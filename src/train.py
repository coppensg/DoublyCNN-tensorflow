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
import utils

from model import Model


def Shape(s):
    return tuple(map(int, s.split(',')))

def parseArgs():
    parser = argparse.ArgumentParser()

    # Code params
    parser.add_argument('-d',   '--dataset',    type=str, default='cifar10')
    parser.add_argument('-s',   '--save_dir',   type=str, default='./save')
    parser.add_argument(        '--load',       action='store_true')
    parser.add_argument('-m',   '--model_file', type=str, default=None)
    parser.add_argument(        '--log',        action='store_true')
    parser.add_argument('-l',   '--path_log',   type=str, default='./logs')

    # Model params
    parser.add_argument('-t', '--conv_type', type=str, default='standard')  # standard or double
    parser.add_argument('-filter_shape', type=Shape, nargs='+', default=[(64, 3, 3), (2, 2)])
    parser.add_argument('-kernel_size', type=int, default=3)
    parser.add_argument('-kernel_pool_size', type=int, default=-1)

    # Training params
    parser.add_argument('-b', '--batch_size', type=int, default=20)
    parser.add_argument('-e', '--train_epochs', type=int, default=150)
    parser.add_argument('-p', '--patience', type=int, default=10)
    parser.add_argument('-lr', type=numpy.float32, default=0.1)
    parser.add_argument('-learning_decay', type=numpy.float32, default=0.5)
    parser.add_argument('-keep_prob', type=str, default=0.5)

    args = parser.parse_args()
    # Save config
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)

    return vars(args)


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


def update_model(sess, model, inputs, target, batch_size, n_batch, keep_prob=1.):
    '''
    Batch processing for the forward and back-prop
    :param sess: current tensorflow session
    :param model: instance of Model
    :param keep_prob: probability to keep a coonection
    '''
    for batch in range(n_batch):
        idx = range(batch * batch_size, (batch + 1) * batch_size)
        input_batch = inputs[idx]
        target_batch = target[idx]

        feed = {model.inputs: input_batch,
                model.targets: target_batch,
                model.keep_prob: keep_prob}

        _ = sess.run([model.train_op], feed)

def fwd_eval(sess, model, inputs, target, batch_size, n_batch):
    '''
    Batch processing for the forward pass
    :param sess: current tensorflow session
    :param model: instance of Model
    :return: [loss, errors]
    '''
    cur_costs, cur_errors = numpy.zeros((n_batch)), numpy.zeros((n_batch))
    for batch in range(n_batch):
        idx = range(batch * batch_size, (batch + 1) * batch_size)
        inputs_batch = inputs[idx]
        target_batch = target[idx]

        feed = {model.inputs: inputs_batch,
                model.targets: target_batch,
                model.keep_prob: 1}
        train_loss, train_err = sess.run([model.loss, model.err], feed)
        cur_costs[batch] = train_loss
        cur_errors[batch] = train_err

    return cur_costs.mean(), cur_errors.mean()


def factory_appender(sess, use_log, log_dir, log_filename):
    '''
    Closure in order to log event when append to a list
    :param sess: current tensorflow session
    :param use_log: bool. if False no logs will be created
    :return: function which append elements to a list
    '''
    if use_log:
        writer = tf.summary.FileWriter(os.path.join(log_dir, log_filename), sess.graph)
    def appender(list, value, epoch, tag):
        list[epoch] = value
        if use_log:
             summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
             writer.add_summary(summary, epoch)
    return appender

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

    (train_x, train_y, train_num) = train
    (valid_x, valid_y, valid_num) = valid

    n_train_batches = train_num/ batch_size
    n_valid_batches = valid_num/ batch_size

    tr_appender = factory_appender(sess=sess, use_log=use_log, log_dir=path_log, log_filename="train")
    va_appender = factory_appender(sess=sess, use_log=use_log, log_dir=path_log, log_filename="valid")

    print "Training..."

    train_errors, valid_errors, train_costs, valid_costs = numpy.zeros((train_epochs)), numpy.zeros((train_epochs)),\
                                                           numpy.zeros((train_epochs)), numpy.zeros((train_epochs))

    best_valid_err = 1.
    best_valid_epoch = 0
    bad_count = 0
    best_sess = sess

    # TODO see if tensorflow use learning rate decay by default
    sess.run(tf.assign(model.lr, lr))
    for epoch in range(train_epochs):
        start = time.time()

        # shuffle the train set
        idx_perm = numpy.random.permutation(train_num)
        train_x = train_x[idx_perm]
        train_y = train_y[idx_perm]

        # compute train loss, err and update weights
        update_model(sess=sess, model=model, inputs=train_x, target=train_y,
                  batch_size=batch_size, n_batch=n_train_batches, keep_prob=keep_prob)

        # Compute training loss and err
        loss, err = fwd_eval(sess=sess, model=model, inputs=train_x, target=train_y,
                              batch_size=batch_size, n_batch=n_train_batches)
        tr_appender(train_errors, err, epoch, "error")
        tr_appender(train_costs, loss, epoch, "loss")

        # compute validation loss and err
        loss, err = fwd_eval(sess=sess, model=model, inputs=valid_x, target=valid_y,
                              batch_size=batch_size, n_batch=n_valid_batches)
        va_appender(valid_errors, err, epoch, "error")
        va_appender(valid_costs, loss, epoch, "loss")


        # keep best model and reduce learning rate if necessary
        if valid_errors[-1] <= best_valid_err:
            best_valid_err = valid_errors[-1]
            best_valid_epoch = epoch
            bad_count = 0
            # update the current best model
            save()
        else:
            bad_count += 1
            if bad_count > patience:
                print 'Reducing the learning rate..'
                sess.run(tf.assign(model.lr, model.lr*learning_decay))
                bad_count = 0

        end = time.time()
        print "epoch. {}, train_loss = {:.4f}, valid_loss = {:.4f}," \
              "train_error = {:.4f}, valid_error = {:.4f}, time/epoch = {:.3f} s" \
            .format(epoch, train_costs[epoch], valid_costs[epoch], train_errors[epoch], valid_errors[epoch], end - start)

        # Not useful anymore
        # # save models
        # if (epoch%10 == 0) and (epoch>0):
        #     save()

    print 'Best errors train {:.4f}, valid {:.4f}'.format(train_errors[best_valid_epoch], valid_errors[best_valid_epoch])



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
    [train, valid, _, num_class, image_shape] = load_normalize_data(dataset)

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
    main(args)

    if args["log"]:
        print ">> tensorboard --logdir={}".format(args['path_log'])


