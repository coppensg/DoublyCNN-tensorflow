import tensorflow as tf
import numpy as np

def init_weights(shape):
	initial = tf.truncated_normal(shape, stedv=stdev_initialization)
	return tf.Varialbe(initial)

def init_biases(shape):
	initial = tf.constant(constant_initialization, shape=shape)
	return tf.Varialbe(initial)

def simple_conv_layer(x, filter_shape, stride=[1,1,1,1], padding='SAME'):
	
	# Define weight and bias shapes and initialize them
	bias_shape = filter_shape[3]
	W = init_weights(filter_shape, name='W')
	bias = init_biases(nb_filter, name='b')

	# Apply convolution
	conv = tf.nn.conv2d(x, W, stride=stride, padding=padding)

	# return activation of convolution + bias terms
	return tf.nn.relu(conv + bias)

def fully_connected_layer(x, num_class):
	dim = x.get_shape()[1].value # number of channels
	W = init_weights(shape=[dim, num_class])
	b = init_biases([num_class])
	return tf.add(tf.matmul(x, W), b, name='final')


class Model:
	def __init__(
		self,
		image_shape,
		filter_shape,
		num_class,
        conv_type,
        kernel_size,
        kernel_pool_size,
        keep_prob,
    ):
    """
    Create instance of a Convolutional Neural Network model
    """

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


        self.n_visible = numpy.prod(image_shape)
        self.n_layers = len(filter_shape)
        self.rng = tf.set_random_seed(123)

        self.conv_layers = []

		keep_prob = tf.placeholder(tf.float32) # include keep_prob in feed_dict


		### build every layers defined in filter_shape ########################################################
		for l in range(n_layers):

			# Convolutional layer case
			if len(filter_shape[l]) == 4:
				
				# Double convolution
				if conv_type == 'double' and filter_shape[l][1] > kernel_size:
					this_layer = double_conv_layer(this_layer, filter_shape[l])
					this_layer = tf.layers.batch_normalization(this_layer)

				# Simple convolution
				elif conv_type == 'standard' or (conv_type == 'double' and filter_shape[l][1] <= kernel_size):
					this_layer = simple_conv_layer(this_layer, filter_shape[l])
					this_layer = tf.layers.batch_normalization(this_layer)

				else:
					raise NotImplementedError

				self.conv_layers.append(this_layer) # Useful ?

			# Pooling layer
			elif len(filter_shape[l]) == 2:
				s = filter_shape[l][0]
				this_layer = tf.nn.max_pool(this_layer, [1, s, s, 1], strides=[1, s, s, 1], padding='SAME')
				this_layer = tf.nn.dropout(this_layer, keep_prob)

			else:
				raise NotImplementedError
		########################################################################################################

		### Global Average Pooling #############################################################################
		########################################################################################################


		### Logits #############################################################################################
		### (use tf.nn.softmax_cross_entropy_with_logits(logits, labels) which is optimized for training #######
		### and tf.nn.softmax(logits) fot prediction) ##########################################################
		flatten = tf.reshape(this_layer, [this_layer.get_shape()[2], -1])
		self.logits_layer = fully_connected_layer(flatten)
		########################################################################################################        	
