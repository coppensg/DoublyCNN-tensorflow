import tensorflow as tf


def double_conv_layer(x, num_filters, filter_size, stride=(1,1), pad=0,
              untie_biases=False, kernel_size=3, kernel_pool_size=1,
              W, b, nonlinearity=lasagne.nonlinearities.rectify,
              flip_filters=True):


    filter_offset = filter_size




