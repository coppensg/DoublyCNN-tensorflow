# Code taken from https://github.com/Shuangfei/doublecnn/blob/master/data/load_cifar100.py
import cPickle as pkl
import numpy as np

data_dir = '../Datasets/cifar-100-python/'
train_file = 'train'
test_file = 'test'

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pkl.load(f)
    dict['data'] = np.asarray(dict['data'], dtype=np.float32)
    return dict

def load_data(data_dir):
    train_data = unpickle(data_dir + train_file)
    test_data = unpickle(data_dir + test_file)

    train_set_x = train_data['data'][:40000] / 255.
    train_set_y = np.array(train_data['fine_labels'][:40000]).astype('int32')

    valid_set_x = train_data['data'][40000:] / 255.
    valid_set_y = np.array(train_data['fine_labels'][40000:]).astype('int32')
    test_set_x = test_data['data'] / 255.
    test_set_y = np.array(test_data['fine_labels']).astype('int32')

    return ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y))


if __name__ == "__main__":
    ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)) = load_data(data_dir)
