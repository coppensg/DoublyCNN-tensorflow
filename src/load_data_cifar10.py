# Code taken from https://github.com/Shuangfei/doublecnn/blob/master/data/load_cifar10.py
import cPickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

data_dir = './Datasets/cifar-10-batches-py/'
train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
valid_file = 'data_batch_5'
test_file = 'test_batch'
train_augmented_file = 'augmented_dataset'

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pkl.load(f)
    dict['data'] = np.asarray(dict['data'], dtype=np.float32)
    return dict

def save_augmented_dataset(data_dir, train_augmented_file, randomize_per_image):
    # Load training data
    train_data = [unpickle(data_dir + file) for file in train_files]
    train_set_x = np.concatenate([batch['data'] for batch in train_data], axis=0) / 255.
    train_set_y = np.concatenate([batch['labels'] for batch in train_data], axis=0).astype('int32')
    

    # For testing
    # train_set_x = train_set_x[:10,:]
    # train_set_y = train_set_y[:10]


    # Generate random distorded transformation of images and concatenate the resulting tensors
    train_set_augmented_x = distord(train_set_x)
    train_set_augmented_y = train_set_y
    for _ in range(randomize_per_image):
        tensor = distord(train_set_x)

        train_set_augmented_x = tf.concat([train_set_augmented_x, tensor], axis=0)
        train_set_augmented_y = np.concatenate([train_set_augmented_y, train_set_y], axis=0)

    # Transform tensors into ndarrays
    with tf.Session() as sess:
        train_set_augmented_x = train_set_augmented_x.eval(session=sess)

    # for i in range(randomize_per_image):
    #     plt.subplot(1,randomize_per_image,i+1)
    #     plt.imshow(train_set_augmented_x[2*i+1].reshape(32,32,3))
    # plt.show()

    augmented_dataset_dict = {'data': train_set_augmented_x, 'labels': train_set_augmented_y}

    # Save pickled augmented dataset
    with open(data_dir+train_augmented_file, 'wb') as f:
        pkl.dump(augmented_dataset_dict, f)

        

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

    return tf.reshape(distorded, [-1, 3*32*32])    


def load_data(datadir):
    train_data = [unpickle(data_dir + file) for file in train_files]
    valid_data = unpickle(data_dir + valid_file)
    test_data = unpickle(data_dir + test_file)

    train_set_x = np.concatenate([batch['data'] for batch in train_data], axis=0) / 255.
    train_set_y = np.concatenate([batch['labels'] for batch in train_data], axis=0).astype('int32')

    valid_set_x = valid_data['data'] / 255.
    valid_set_y = np.array(valid_data['labels']).astype('int32')

    test_set_x = test_data['data'] / 255.
    test_set_y = np.array(test_data['labels']).astype('int32')

    return ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y))


def load_data_augmented(datadir):
    train_data = unpickle(data_dir + train_augmented_file)
    valid_data = unpickle(data_dir + valid_file)
    test_data = unpickle(data_dir + test_file)

    train_set_x = train_data['data']# / 255.
    train_set_y = train_data['labels']

    valid_set_x = valid_data['data'] / 255.
    valid_set_y = np.array(valid_data['labels']).astype('int32')

    test_set_x = test_data['data'] / 255.
    test_set_y = np.array(test_data['labels']).astype('int32')

    return ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y))    


if __name__ == "__main__":
    print "Uncomment save_augmented_dataset() line to create the augmented dataset"
    # start = time.time()
    # save_augmented_dataset(data_dir, train_augmented_file, randomize_per_image=9)
    # end = time.time()

    # print "time/10images = {:.3f} s".format(end-start)
    # ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)) = load_data_augmented(data_dir)
    # nb_im = train_set_x.shape[0]
    # for i in range(nb_im):
    #     plt.subplot(1,nb_im,i+1)
    #     plt.imshow(train_set_x[i].reshape(32,32,3))
    # plt.show()