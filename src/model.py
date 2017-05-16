import tensorflow as tf
import numpy
import re


def init_weights(shape, name="W"):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer(uniform=True))


def init_biases(constant, shape, name="b"):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.constant_initializer(constant))

def build_extractor(W_effective_shape):
    return tf.reshape(tf.Variable(tf.diag(tf.ones(numpy.prod(W_effective_shape[0:3])))),
                            W_effective_shape[0:3]+[numpy.prod(W_effective_shape[0:3]),])

def filter_to_image(W):
    return tf.transpose(W, perm=[3, 0, 1, 2])

def image_to_filter(W,W_shape):
    W = tf.transpose(W, perm=[3, 0, 1, 2])
    W = tf.reshape(W, W_shape)
    return W

def reshape_before_pooling(output, width, height, filter_offset, num_filters):
    output = tf.transpose(output, [0, 3, 1, 2])
    return tf.reshape(output, [-1, filter_offset, filter_offset, width*height*num_filters])

def reshape_after_pooling(I, width, height, num_filters):
    depth = int(I.get_shape()[1].value)
    depth = depth*depth*num_filters
    I = tf.transpose(I, [0, 3, 1, 2])
    return tf.reshape(I, [-1, width, height, depth])

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
        W = init_weights(shape=[dim, num_class], name='W') # default in layers.DenseLayer is Glorotuniform
        # W = tf.get_variable(name="W_fully_connected", shape=[dim, num_class],
        # initializer=tf.constant_initializer(W_fully_connected_const)) # Initialized with constant 0. in the code line 187
        b = init_biases(bias_const, [num_class], name='b')
        return tf.add(tf.matmul(x, W), b, name=name)


def double_conv_layer(x, filter_shape,
                      kernel_size=3, kernel_pool_size=1,
                      nonlinearity=tf.nn.relu, padding='SAME', name='double_conv_layer'):

        # conv2d
        # input shape [batch, in_height, in_width, in_channels]
        # filter shape [filter_height, filter_width, in_channels, out_channels]
    with tf.variable_scope(name):
        
        bias_const = 0.1

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
        W_effective = tf.nn.conv2d(W_meta, extractor, strides=[1,1,1,1], padding='VALID')

        # Second convolution : convolve effective filters to images
        W_effective = image_to_filter(W_effective, W_effective_shape)
        output = tf.nn.conv2d(x, W_effective, strides=[1, 1, 1, 1], padding=padding)
        
        # Add bias and apply non-linearity
        bias = init_biases(bias_const, [W_effective.get_shape()[3].value])
        output = nonlinearity(output + bias)

        # Reorganise output for pooling
        next_width = int(output.get_shape()[1].value)
        next_height = int(output.get_shape()[2].value)
        output = reshape_before_pooling(output, next_width, next_height, filter_offset, num_filters)
        
        # Pooling
        I = tf.nn.max_pool(output, ksize=[1, kernel_pool_size, kernel_pool_size, 1], strides=[1, kernel_pool_size, kernel_pool_size, 1], padding='SAME')
        
        # Reorganise output to image
        I = reshape_after_pooling(I, next_width, next_height, num_filters)

        return I















def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  TOWER_NAME = "tower"
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))







def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv_layer(img, shape=[3,3,128,128], n=1):
    # conv_n
    with tf.variable_scope('conv'+str(n)) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=shape,
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(img, kernel, [1, 1, 1, 1], padding='SAME')

        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)

        norm = tf.layers.batch_normalization(pre_activation, name='norm'+str(n))

        activ = tf.nn.relu(norm, name=scope.name)
        _activation_summary(activ)

        return activ


def inference(images, keep_prob):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  conv1 = conv_layer(images, shape=[3,3,3,128], n=1)
  conv2 = conv_layer(conv1, shape=[3,3,128,128], n=2)


  # pool1
  pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool1')
  pool1 = tf.nn.dropout(pool1, keep_prob=keep_prob, name='Dropout_1')


  conv3 = conv_layer(pool1, shape=[3,3,128,128], n=3)
  conv4 = conv_layer(conv3, shape=[3,3,128,128], n=4)


  # pool2
  pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool2')
  pool2 = tf.nn.dropout(pool2, keep_prob=keep_prob, name='Dropout_2')


  conv5 = conv_layer(pool2, shape=[3,3,128,128], n=5)
  conv6 = conv_layer(conv5, shape=[3,3,128,128], n=6)


  # pool3
  pool3 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool3')
  pool3 = tf.nn.dropout(pool3, keep_prob=keep_prob, name='Dropout_3')


  conv7 = conv_layer(pool3, shape=[3,3,128,128], n=7)
  conv8 = conv_layer(conv7, shape=[3,3,128,128], n=8)



  # pool3
  pool4 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4')
  pool4 = tf.nn.dropout(pool4, keep_prob=keep_prob, name='Dropout_4')

  cur_layer = tf.reduce_mean(pool4, reduction_indices=[3], keep_dims=True)

  fc_input_size = int(cur_layer.get_shape()[1] * cur_layer.get_shape()[2] * cur_layer.get_shape()[3])
  flatten = tf.reshape(cur_layer, [-1, fc_input_size])





  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [1024, 10],
                                          stddev=1/1024.0, wd=0.0)
    biases = _variable_on_cpu('biases', [10],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(flatten, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear







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
        self.inputs = tf.placeholder(tf.float32, [None, 32,32,3], name="input")
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

        
        # ### Build every layers defined in filter_shape ########################################################
        # for l in range(self.n_layers):
        #
        #     # Convolutional layer case
        #     if len(filter_shape[l]) == 3:
        #         out_depth, height, width = filter_shape[l]
        #         shape = [height, width, in_depth, out_depth]
        #         # Double convolution
        #         if conv_type == 'double' and filter_shape[l][1] > kernel_size:
        #             print "Building double conv layer, shape : " + str(filter_shape[l][1])
        #             cur_layer = double_conv_layer(cur_layer, shape, kernel_size=kernel_size,
        #                                           kernel_pool_size=kernel_pool_size, padding='SAME',
        #                                           name='DoublyCNN2D_{}'.format(l))
        #             cur_layer = tf.layers.batch_normalization(cur_layer, name='DoublyCNN2D_{}_BN'.format(l))
        #         # Simple convolution
        #         elif conv_type == 'standard' or (conv_type == 'double' and filter_shape[l][1] <= kernel_size):
        #             print "Building simple conv layer, shape : " + str(filter_shape[l][1])
        #             cur_layer = simple_conv_layer(cur_layer, shape, strides=[1,1,1,1], padding='SAME',
        #                                           name='CNN2D_{}'.format(l))
        #             cur_layer = tf.layers.batch_normalization(cur_layer, name='CNN2D_{}_BN'.format(l))
        #         else:
        #             raise NotImplementedError
        #         in_depth = out_depth
        #
        #         cur_layer = tf.nn.relu(cur_layer, name='Relu_{}'.format(l))
        #
        #     # Pooling layer
        #     elif len(filter_shape[l]) == 2:
        #         s = filter_shape[l][0]
        #         cur_layer = tf.nn.max_pool(cur_layer, [1, s, s, 1], strides=[1, s, s, 1], padding='SAME', name='Pool_{}'.format(l))
        #         cur_layer = tf.nn.dropout(cur_layer, keep_prob=self.keep_prob, name='Dropout_{}'.format(l))
        #     else:
        #         raise NotImplementedError
        #
        #     self.layers.append(cur_layer)
        #
        # ########################################################################################################

        
        ### Global Average Pooling #############################################################################
        
        ## Average pooling not implemented for non-spatial dimensions ##
        # avg_pool_input_depth = int(cur_layer.get_shape()[3])
        # cur_layer = tf.nn.avg_pool(cur_layer, ksize=[1, 1, 1, avg_pool_input_depth], strides=[1, 1, 1, avg_pool_input_depth], padding='SAME', name='GAPL')
        # cur_layer = tf.layers.average_pool2d(cur_layer, ksize=[1, 1], strides=[1, avg_pool_input_depth], padding='SAME', name='GAPL')

        ## Do it with tf.reduce_mean(axis=3) ##
        # cur_layer = tf.reduce_mean(cur_layer, reduction_indices=[3], keep_dims=True)
        # ########################################################################################################
        #
        #
        # ### Logits #############################################################################################
        # ### (use tf.nn.softmax_cross_entropy_with_logits(logits, labels) which is optimized for training #######
        # ### and tf.nn.softmax(logits) fot prediction) ##########################################################
        # fc_input_size = int(cur_layer.get_shape()[1]*cur_layer.get_shape()[2]*cur_layer.get_shape()[3])
        # flatten = tf.reshape(cur_layer, [-1, fc_input_size])
        # self.logits = fully_connected_layer(flatten, num_class, name="FCL")
        ########################################################################################################
        self.logits = inference(self.inputs, keep_prob=self.keep_prob)

        self.probs = tf.nn.softmax(self.logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))

        uncorrect_prediction = tf.not_equal(tf.argmax(self.probs, 1), tf.argmax(self.targets, 1))
        self.err = tf.reduce_mean(tf.cast(uncorrect_prediction, tf.float32))

        self.lr = tf.Variable(0.0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
