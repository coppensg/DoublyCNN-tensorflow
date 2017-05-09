import tensorflow as tf
import numpy as np


def double_conv_layer(x, filter_shape,
                      kernel_size=3, kernel_pool_size=1,
                      nonlinearity=tf.nn.relu, padding='SAME', name='double_conv_layer'):

    filter_size, filter_size, filter_depth, num_filters = filter_shape

    with tf.variable_scope(name):
        filter_offset = filter_size - kernel_size + 1
        n_times = filter_offset ** 2

        # sub filter W
        W_shape = [kernel_size, kernel_size, filter_depth, num_filters * n_times]

        # W_meta = init_weigths(filter_shape, name="W_meta")
        # b_meta = init_biases([num_filters], name="b_meta")
        W_meta = tf.get_variable("W", filter_shape,
                            initializer=tf.random_normal_initializer(0, 0.1))
        b_meta = tf.get_variable("b", [num_filters],
                        initializer=tf.constant_initializer(0.1))


        tf.set_random_seed(123) # todo voir si c'est utile


        filter = tf.reshape(tf.Variable(tf.diag(tf.ones(np.prod(W_shape[0:3])))),
                            W_shape[0:3]+[np.prod(W_shape[0:3]),])
        W_meta = tf.transpose(W_meta, perm=[3,0,1,2])
        # conv2d
        # input shape [batch, in_height, in_width, in_channels]
        # filter shape [filter_height, filter_width, in_channels, out_channels]
        W_effective = tf.nn.conv2d(W_meta, filter, strides=[1,1,1,1], padding='VALID')

        W_effective = tf.transpose(W_effective, perm=[3, 0, 1, 2])
        W_effective = tf.reshape(W_effective, W_shape)

        output = tf.nn.conv2d(x, W_effective, strides=[1, 1, 1, 1], padding=padding)
        # todo voir s'il faut ajouter un biais

        output = nonlinearity(output)

        batch_size = tf.shape(x)[0]
        width = tf.shape(x)[1]
        height = tf.shape(x)[2]
        output = tf.transpose(output, [0, 3, 1, 2])
        output = tf.reshape(output, [batch_size, filter_offset, filter_offset, width*height*num_filters])

        I = tf.nn.max_pool(output, ksize=[1, kernel_pool_size, kernel_pool_size, 1], strides=[1, 1, 1, 1], padding='SAME')

        I = tf.transpose(I, [0, 2, 3, 1])
        I = tf.reshape(I, [batch_size, width, height, -1])
        return I


def test_mnist():

    def conv_layer(input_, shape_W, shape_b, strides, name='conv_layer'):
        with tf.variable_scope(name):
            W = tf.get_variable("W", shape_W,
                                initializer=tf.random_normal_initializer(0, 0.1))
            b = tf.get_variable("b", shape_b,
                                initializer=tf.constant_initializer(0.1))

            conv2D = tf.nn.conv2d(input_, W, strides=strides, padding='SAME') + b
        # return tf.nn.relu(conv2D + b)
        return conv2D


    def full_connected_layer(input_, shape_W, shape_b, activation_func=None, name='fc_layer'):
        with tf.variable_scope(name):
            init_W = tf.truncated_normal(shape_W, stddev=0.1)
            init_b = tf.constant(0.1, shape=shape_b)

            W = tf.Variable(init_W, name="W")
            b = tf.Variable(init_b, name="b")

            Y = tf.matmul(input_, W) + b
        if activation_func == 'softmax':
            return tf.nn.softmax(Y)
        elif activation_func == 'relu':
            return tf.nn.relu(Y)
        elif activation_func == 'sigmoid':
            return tf.nn.sigmoid(Y)
        else:
            return Y

    class ModelDCNN():

        def __init__(self):

            # 1. Def variables, placeholders
            self.targets = tf.placeholder(tf.float32, [None, 10], name="label")
            self.inputs = tf.placeholder(tf.float32, [None, 784], name="input")
            formated_input = tf.reshape(self.inputs, [-1, 28, 28, 1], name="input_formatted")

            # 2. def hidden layers
            # conv layer 1
            # conv1 = conv_layer(formated_input,
            #                    shape_W=[5, 5, 1, 4],
            #                    shape_b=[4],
            #                    strides=[1, 1, 1, 1],
            #                    name="conv_layer_1")
            conv1 = double_conv_layer(formated_input, [5,5,1,4],
                      kernel_size=5, kernel_pool_size=2,
                      nonlinearity=tf.nn.relu, padding='SAME', name="conv1")

            # conv 2
            # conv2 = double_conv_layer(conv1, [5,5,4,8],
            #           kernel_size=5, kernel_pool_size=2,
            #           nonlinearity=tf.nn.relu, padding='SAME', name="conv2")
            #print conv2
            conv2 = conv_layer(conv1,
                               shape_W=[5, 5, 4, 8],
                               shape_b=[8],
                               strides=[1, 2, 2, 1],
                               name="conv_layer_2")
            # conv 3
            conv3 = conv_layer(conv2,
                               shape_W=[4, 4, 8, 12],
                               shape_b=[12],
                               strides=[1, 2, 2, 1],
                               name="conv_layer_3")

            # change the shape of conv3
            fc1_input_size = 7 * 7 * 12
            conv3_flat = tf.reshape(conv3, [-1, fc1_input_size], name="conv3_flat")
            # full connected layer 1
            fc_1 = full_connected_layer(conv3_flat,
                                        shape_W=[fc1_input_size, 200],
                                        shape_b=[200],
                                        activation_func='relu',
                                        name="fc_1")
            # full connected layer 2
            self.logits = full_connected_layer(fc_1,
                                               shape_W=[200, 10],
                                               shape_b=[10],
                                               name="output_layer")
            self.probs = tf.nn.softmax(self.logits)

            # 3. Def the loss func
            # self.loss = tf.reduce_mean(-tf.reduce_sum(self.targets*tf.log(self.probs), reduction_indices=[1]))
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(self.probs), reduction_indices=[1]))
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))
            # 4. Def accuracy
            correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.targets, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # 5. Train
            l_rate = 0.001
            self.train_op = tf.train.AdamOptimizer(l_rate).minimize(self.loss)

    # train_step
    def training_step(sess, model, i, update_train, update_test):
        # print(i)

        # read the current batch
        X_batch, Y_batch = mnist.train.next_batch(100)
        sess.run(model.train_op, feed_dict={model.inputs: X_batch, model.targets: Y_batch})

        ## model evaluation
        train_a = []
        train_c = []
        test_a = []
        test_c = []
        if update_train:
            a, c = sess.run([model.accuracy, model.loss], feed_dict={model.inputs: X_batch, model.targets: Y_batch})
            train_a.append(a)
            train_c.append(c)

        if update_test:
            a, c = sess.run([model.accuracy, model.loss],
                            feed_dict={model.inputs: mnist.test.images, model.targets: mnist.test.labels})
            test_a.append(a)
            test_c.append(c)

        return (train_a, train_c, test_a, test_c)


    # import mnist dataset
    from tensorflow.examples.tutorials.mnist import input_data

    # load dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    train_a = []
    train_c = []
    test_a = []
    test_c = []

    training_iter = 3000
    epoch_size = 100

    # initialize sess
    model = ModelDCNN()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(training_iter):
            # test = False
            # if i % epoch_size == 0:
            test = True
            a, c, ta, tc = training_step(sess, model, i, test, test)
            train_a += a
            train_c += c
            test_a += ta
            test_c += tc
            print "Test acc = " + str(ta)




if __name__ == "__main__":
    test_mnist()