import tensorflow as tf
import numpy


def init_weights(shape, name="W"):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer(uniform=True))


def init_biases(constant, shape, name="b"):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.constant_initializer(constant))


def build_extractor(W_effective_shape):
    return tf.reshape(
        tf.Variable(tf.diag(tf.ones(numpy.prod(W_effective_shape[0:3]))), trainable=False),
        W_effective_shape[0:3] + [numpy.prod(W_effective_shape[0:3]), ])


def filter_to_image(W):
    return tf.transpose(W, perm=[3, 0, 1, 2])


def image_to_filter(W, W_shape):
    W = tf.transpose(W, perm=[3, 0, 1, 2])
    W = tf.reshape(W, W_shape)
    return W


def reshape_before_pooling(output, width, height, filter_offset, num_filters):
    output = tf.transpose(output, [0, 3, 1, 2])
    return tf.reshape(output, [-1, filter_offset, filter_offset, width * height * num_filters])


def reshape_after_pooling(I, width, height, num_filters):
    depth = int(I.get_shape()[1].value)
    depth = depth * depth * num_filters
    I = tf.transpose(I, [0, 3, 1, 2])
    return tf.reshape(I, [-1, width, height, depth])


def simple_conv_layer(x, filter_shape, strides=[1, 1, 1, 1], padding='SAME', name="CNN2D"):
    bias_const = 0.

    with tf.variable_scope(name):
        # Define weight and bias shapes and initialize them
        W = init_weights(filter_shape, name='W')
        bias = init_biases(bias_const, [filter_shape[3]], name='b')
        # Apply convolution
        conv = tf.nn.conv2d(x, W, strides=strides, padding=padding)
        # Apply batch normalization
        output = tf.layers.batch_normalization(conv + bias)
        # apply non - linearity
        output = tf.nn.relu(output)
        # return activation of convolution + bias terms
        return output


def fully_connected_layer(x, num_class, name="FCL"):
    bias_const = 0.
    with tf.variable_scope(name):
        dim = x.get_shape()[1].value  # number of channels
        W = init_weights(shape=[dim, num_class], name='W')
        b = init_biases(bias_const, [num_class], name='b')
        return tf.add(tf.matmul(x, W), b, name=name)


def double_conv_layer(x, filter_shape,
                      kernel_size=3, kernel_pool_size=1,
                      nonlinearity=tf.nn.relu, padding='SAME', name='double_conv_layer'):
    # conv2d
    # input shape [batch, in_height, in_width, in_channels]
    # filter shape [filter_height, filter_width, in_channels, out_channels]
    with tf.variable_scope(name):
        bias_const = 0.

        # Define all shapes        
        filter_size, filter_size, filter_depth, num_filters = filter_shape
        filter_offset = filter_size - kernel_size + 1
        n_times = filter_offset ** 2
        W_effective_shape = [kernel_size, kernel_size, filter_depth, num_filters * n_times]

        # Initalize meta filters
        W_meta = init_weights(filter_shape, name="W_meta")

        # First convolution : extract effective filters
        extractor = build_extractor(W_effective_shape)
        W_meta = filter_to_image(W_meta)
        W_effective = tf.nn.conv2d(W_meta, extractor, strides=[1, 1, 1, 1], padding='VALID')

        # Second convolution : convolve effective filters to images
        W_effective = image_to_filter(W_effective, W_effective_shape)
        output = tf.nn.conv2d(x, W_effective, strides=[1, 1, 1, 1], padding=padding)

        # Add bias
        bias = init_biases(bias_const, [W_effective.get_shape()[3].value])
        # batch normalization
        output = tf.layers.batch_normalization(output + bias)
        # apply non - linearity
        output = nonlinearity(output)

        # Reorganise output for pooling
        next_width = int(output.get_shape()[1].value)
        next_height = int(output.get_shape()[2].value)
        output = reshape_before_pooling(output, next_width, next_height, filter_offset, num_filters)

        # Pooling
        I = tf.nn.max_pool(output, ksize=[1, kernel_pool_size, kernel_pool_size, 1],
                           strides=[1, kernel_pool_size, kernel_pool_size, 1], padding='SAME')

        # Reorganise output to image
        I = reshape_after_pooling(I, next_width, next_height, num_filters)

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
        reshaped_inputs = tf.reshape(self.inputs, [-1, in_depth, height, width],
                                     name="input_formatted")
        reshaped_inputs = tf.transpose(reshaped_inputs, perm=[0, 2, 3, 1])

        self.filter_shape = filter_shape
        self.n_layers = len(filter_shape)
        # todo voir si necessaire
        # self.rng = tf.set_random_seed(123)
        self.layers = []
        self.layers.append(reshaped_inputs)
        self.keep_prob = tf.placeholder(tf.float32)  # include keep_prob in feed_dict

        ### Build every layers defined in filter_shape ########################################################
        for l in range(self.n_layers):

            # Convolutional layer case
            if len(filter_shape[l]) == 3:
                out_depth, height, width = filter_shape[l]
                # Double convolution
                if conv_type == 'double' and filter_shape[l][1] > kernel_size:
                    print "Building double conv layer, shape : " + str(filter_shape[l][1])
                    shape = [height, width, in_depth, out_depth]
                    self.layers.append(
                        double_conv_layer(self.layers[-1], shape, kernel_size=kernel_size,
                                          kernel_pool_size=kernel_pool_size, padding='SAME',
                                          name='DoublyCNN2D_{}'.format(l)))
                # Simple convolution
                elif conv_type == 'standard' or (
                        conv_type == 'double' and filter_shape[l][1] <= kernel_size):
                    if filter_shape[l][1] <= kernel_size:
                        height, width = kernel_size, kernel_size
                    shape = [height, width, in_depth, out_depth]
                    print shape
                    print "Building simple conv layer, shape : " + str(filter_shape[l][1])
                    self.layers.append(
                        simple_conv_layer(self.layers[-1], shape, strides=[1, 1, 1, 1],
                                          padding='SAME',
                                          name='CNN2D_{}'.format(l)))
                else:
                    raise NotImplementedError
                in_depth = out_depth
            # Pooling layer
            elif len(filter_shape[l]) == 2:
                s = filter_shape[l][0]
                self.layers.append(
                    tf.nn.max_pool(self.layers[-1], [1, s, s, 1], strides=[1, s, s, 1],
                                   padding='SAME', name='Pool_{}'.format(l)))
                self.layers.append(tf.nn.dropout(self.layers[-1], keep_prob=self.keep_prob,
                                                 name='Dropout_{}'.format(l)))
            else:
                raise NotImplementedError
        ########################################################################################################


        ### Global Average Pooling #############################################################################
        if len(filter_shape[self.n_layers - 1]) != 3 or filter_shape[self.n_layers - 1][
            0] != num_class:
            print filter_shape[self.n_layers - 1]
            print filter_shape[self.n_layers - 1][0]
            raise ValueError(
                "The last layer must be a convolution layer with {} filters.".format(num_class))

        height, width = map(int, list(self.layers[-1].get_shape())[1:-1])
        self.layers.append(tf.nn.avg_pool(self.layers[-1], ksize=[1, height, width, 1],
                                          strides=[1, height, width, 1], padding='VALID',
                                          name='Global_AVG_POOL'))

        ### Logits #############################################################################################
        ### (use tf.nn.softmax_cross_entropy_with_logits(logits, labels) which is optimized for training #######
        ### and tf.nn.softmax(logits) fot prediction) ##########################################################

        last_layer_shape = map(int, list(self.layers[-1].get_shape())[1:])
        self.logits = tf.reshape(self.layers[-1], [-1, numpy.prod(last_layer_shape)])
        self.probs = tf.nn.softmax(self.logits)

        # loss
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))

        # error
        uncorrect_prediction = tf.not_equal(tf.argmax(self.probs, 1), tf.argmax(self.targets, 1))
        self.err = tf.reduce_mean(tf.cast(uncorrect_prediction, tf.float32))

        # optimizer
        self.lr = tf.Variable(1.0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
