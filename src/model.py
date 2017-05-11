import tensorflow as tf
import numpy


def init_weights(shape, name="W"):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer(uniform=True))


def init_biases(constant, shape, name="b"):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.constant_initializer(constant))


def simple_conv_layer(x, filter_shape, strides=[1, 1, 1, 1], padding='SAME', name="CNN2D"):
    # todo voir comment initialiser les params
    bias_const = 0

    with tf.variable_scope(name):
        # Define weight and bias shapes and initialize them
        W = init_weights(filter_shape, name='W')
        bias = init_biases(bias_const, [filter_shape[3]], name='b')

        # Apply convolution
        conv = tf.nn.conv2d(x, W, strides=strides, padding=padding)

        # return activation of convolution + bias terms
        return tf.nn.relu(conv + bias)


def fully_connected_layer(x, num_class, name="FCL"):
    # todo voir comment initialiser les params
    bias_const = 0
    W_fully_connected_const = 0
    with tf.variable_scope(name):
        dim = x.get_shape()[1].value  # number of channels
        # W = init_weights(shape=[dim, num_class], name='W') # default in layers.DenseLayer is Glorotuniform
        W = tf.get_variable(name="W_fully_connected", shape=[dim, num_class], initializer=tf.constant_initializer(W_fully_connected_const)) # Initialized with constant 0. in the code line 187
        b = init_biases(bias_const, [num_class], name='b')
        return tf.add(tf.matmul(x, W), b, name=name)


def double_conv_layer(x, filter_shape,
                      kernel_size=3, kernel_pool_size=1,
                      nonlinearity=tf.nn.relu, padding='SAME', name='double_conv_layer'):

    filter_size, filter_size, filter_depth, num_filters = filter_shape

    with tf.variable_scope(name):
        bias_const = 0.1

        filter_offset = filter_size - kernel_size + 1
        n_times = filter_offset ** 2

        # sub filter W
        W_shape = [kernel_size, kernel_size, filter_depth, num_filters * n_times]

        W_meta = init_weights(filter_shape, name="W_meta")
        b_meta = init_biases(bias_const, [num_filters], name="b_meta")

        filter = tf.reshape(tf.Variable(tf.diag(tf.ones(numpy.prod(W_shape[0:3])))),
                            W_shape[0:3]+[numpy.prod(W_shape[0:3]),])

        W_meta = tf.transpose(W_meta, perm=[3,0,1,2])
        # conv2d
        # input shape [batch, in_height, in_width, in_channels]
        # filter shape [filter_height, filter_width, in_channels, out_channels]
        W_effective = tf.nn.conv2d(W_meta, filter, strides=[1,1,1,1], padding='VALID')

        W_effective = tf.transpose(W_effective, perm=[3, 0, 1, 2])
        W_effective = tf.reshape(W_effective, W_shape)
        output = tf.nn.conv2d(x, W_effective, strides=[1, 1, 1, 1], padding=padding)
        # todo voir s'il faut ajouter un biais
        #output = nonlinearity(output)

        width = int(output.get_shape()[1].value)
        height = int(output.get_shape()[2].value)

        output = tf.transpose(output, [0, 3, 1, 2])
        output = tf.reshape(output, [-1, filter_offset, filter_offset, width*height*num_filters])
        print output
        I = tf.nn.max_pool(output, ksize=[1, kernel_pool_size, kernel_pool_size, 1], strides=[1, kernel_pool_size, kernel_pool_size, 1], padding='SAME')
        depth = int(I.get_shape()[1].value)
        depth = depth*depth*num_filters
        I = tf.transpose(I, [0, 3, 1, 2])
        I = tf.reshape(I, [-1, width, height, depth])

        return I




class Model:
    def __init__(
        self,
        image_shape,
        filter_shape,
        num_class,
        conv_type,
        kernel_size,
        kernel_pool_size,
    ):
        """
        Create instance of a Convolutional Neural Network model
        """
        self.targets = tf.placeholder(tf.float32, [None, num_class], name="label")
        self.inputs = tf.placeholder(tf.float32, [None, numpy.prod(image_shape)], name="input")
        in_depth, height, width = image_shape
        cur_layer = tf.reshape(self.inputs, [-1, in_depth, height, width], name="input_formatted")
        cur_layer = tf.transpose(cur_layer, perm=[0,2,3,1])

        self.filter_shape = filter_shape
        # filter_shape has the form
        #
        #		[
        #
        #		 	[filter_size, filter_size, nb_channel, nb_filter] ,
        #
        #		 	[filter_size, filter_size, nb_channel, nb_filter] ,
        #
        #			[pool-s, pool-s] ,
        #
        #		 	... n_layers times ... ,
        #
        #			... ,
        #
        #		 	[filter_size, filter_size, nb_channel, nb_filter]
        #
        #		]
        #

        self.n_layers = len(filter_shape)
        # todo voir si necessaire
        self.rng = tf.set_random_seed(123)
        self.layers = []

        self.keep_prob = tf.placeholder(tf.float32)  # include keep_prob in feed_dict

        
        ### Build every layers defined in filter_shape ########################################################
        for l in range(self.n_layers):

            # Convolutional layer case
            if len(filter_shape[l]) == 3:
                out_depth, height, width = filter_shape[l]
                shape = [height, width, in_depth, out_depth]
                # Double convolution
                if conv_type == 'double' and filter_shape[l][1] > kernel_size:
                    cur_layer = double_conv_layer(cur_layer, shape, kernel_size=kernel_size,
                                                  kernel_pool_size=kernel_pool_size, padding='SAME',
                                                  name='DoublyCNN2D_{}'.format(l))
                    cur_layer = tf.layers.batch_normalization(cur_layer, name='DoublyCNN2D_{}_BN'.format(l))
                # Simple convolution
                elif conv_type == 'standard' or (conv_type == 'double' and filter_shape[l][1] <= kernel_size):
                    cur_layer = simple_conv_layer(cur_layer, shape, strides=[1,1,1,1], padding='SAME',
                                                  name='CNN2D_{}'.format(l))
                    cur_layer = tf.layers.batch_normalization(cur_layer, name='CNN2D_{}_BN'.format(l))
                else:
                    raise NotImplementedError
                in_depth = out_depth
            # Pooling layer
            elif len(filter_shape[l]) == 2:
                s = filter_shape[l][0]
                cur_layer = tf.nn.max_pool(cur_layer, [1, s, s, 1], strides=[1, s, s, 1], padding='SAME', name='Pool_{}'.format(l))
                cur_layer = tf.nn.dropout(cur_layer, keep_prob=self.keep_prob, name='Dropout_{}'.format(l))
            else:
                raise NotImplementedError

            self.layers.append(cur_layer)

        ########################################################################################################

        
        ### Global Average Pooling #############################################################################

        ########################################################################################################


        ### Logits #############################################################################################
        ### (use tf.nn.softmax_cross_entropy_with_logits(logits, labels) which is optimized for training #######
        ### and tf.nn.softmax(logits) fot prediction) ##########################################################
        fc_input_size = int(cur_layer.get_shape()[1]*cur_layer.get_shape()[2]*cur_layer.get_shape()[3])
        flatten = tf.reshape(cur_layer, [-1, fc_input_size])
        self.logits = fully_connected_layer(flatten, num_class, name="FCL")
        ########################################################################################################

        self.probs = tf.nn.softmax(self.logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))

        uncorrect_prediction = tf.not_equal(tf.argmax(self.probs, 1), tf.argmax(self.targets, 1))
        self.err = tf.reduce_mean(tf.cast(uncorrect_prediction, tf.float32))

        self.lr = tf.Variable(0.0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
