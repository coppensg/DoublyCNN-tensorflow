import sys
import os

import numpy
import tensorflow as tf

import load_data_cifar10
import load_data_cifar100


def restore_model(saver, session, loadfrom):
    saver.restore(session, loadfrom)
    print "Model restored from " + loadfrom

def store_model(saver, session, saveto):
    # Tricky closure
    print saveto
    def save(sess=session):
        if saveto is not None:
            save_path = saver.save(sess, saveto)
            print "Model saved to " + save_path
    return save

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



def distord(inputs):
    num_inputs = inputs.shape[0]
    inputs = inputs.reshape(inputs.shape[0],3,32,32).transpose(0,2,3,1)
    
    padding_fn = lambda x: tf.image.resize_image_with_crop_or_pad(x, 40, 40)
    random_flip_fn = lambda x: tf.image.random_flip_left_right(x)
    random_brightness_fn = lambda x: tf.image.random_brightness(x, max_delta=0.3)
    random_contrast_fn = lambda x: tf.image.random_contrast(x, lower=0.2, upper=1.4)

    distorded = tf.map_fn(padding_fn, inputs)
    distorded = tf.random_crop(distorded, [num_inputs, 32,32,3])
    distorded = tf.map_fn(random_flip_fn, distorded)    
    distorded = tf.map_fn(random_brightness_fn, distorded)

    distorded = tf.map_fn(random_contrast_fn, distorded)

    distorded = tf.minimum(distorded, 1.)
    distorded = tf.maximum(distorded, 0.)

    distorded = tf.transpose(distorded, perm=[0, 3, 1, 2])

    return tf.reshape(distorded, [-1, 3*32*32])