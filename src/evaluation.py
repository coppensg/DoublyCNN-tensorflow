import argparse
import sys
import os
from six.moves import cPickle
import tensorflow as tf
import numpy

import load_data_cifar10
import load_data_cifar100

import utils
from model import Model

def parseArgs():

    parser = argparse.ArgumentParser()

    # Code params
    parser.add_argument('-s', '--save_dir', type=str, required=True)

    args = parser.parse_args()
    return args

def main(arg0):

    # Load Parameters of the model from save_dir/config.pkl
    print "Load parameters of the model..."
    with open(os.path.join(arg0.save_dir, 'config.pkl'), 'rb') as f:
        args = cPickle.load(f)
    arg = vars(args)
    # Program Parameters
    dataset = arg['dataset']
    print 'dataset : {}'.format(dataset)
    loadfrom = os.path.join(arg0.save_dir, arg['model_file'])

    # Topology model
    conv_type = arg['conv_type']
    filter_shape = arg['filter_shape']
    kernel_size = arg['kernel_size']
    kernel_pool_size = arg['kernel_pool_size']
    print 'conv_type : {}'.format(conv_type)
    print 'filter_shape : {}'.format(filter_shape)
    if conv_type == 'double':
        print 'kernel_size : {}'.format(kernel_size)
        print 'kernel_pool_size : {}'.format(kernel_size)



    # Model data
    batch_size = arg['batch_size']

    # Data
    [_, _,test, num_class, image_shape] = utils.load_normalize_data(dataset)
    (test_x, test_y, test_num) = test

    # Topology
    model = Model(image_shape, filter_shape, num_class, conv_type, kernel_size, kernel_pool_size)


    with tf.Session() as sess:
        print "Loading model..."
        saver = tf.train.Saver()
        utils.restore_model(saver, sess, loadfrom)

        print "Compute error on test set..."
        n_test_batches = test_num/ batch_size
        _, err = utils.fwd_eval(sess, model, test_x, test_y, batch_size, n_test_batches)
        print 'The error on test set of {} is {:.4f}'.format(dataset,err)



if __name__ == "__main__":
    args = parseArgs()
    main(args)