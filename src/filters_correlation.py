import numpy as np

def k_translation_correlation(k, W1, W2):
    '''
    k-translation correlation between two convolutional filter within the same layer

    :param k: maximum translation in x-axis and y-axis of W2
    :param W1: array_like (depth, width, height), convolutional filter 1
    :param W2: array_like (depth, width, height), convolutional filter 2
    :return: k-translation correlation of W1 and W2
    '''
    depth, width, height = W1.shape

    rho_k = 0
    W1_flat = np.reshape(W1, depth*width*height, 1)
    W2_pad = np.zeros((depth, width+2*k, height+2*k))
    W2_pad[:,k:k+width,k:k+height] = W2

    for x in range(-k,k+1):
        for y in range(-k,k+1):
            if x == 0 and y == 0:
                continue
            W2_flat = np.reshape(W2_pad[:,k+x:k+width+x,k+y:k+height+y], depth*width*height, 1)
            tmp = np.dot(W1_flat.T,W2_flat)

            if tmp > rho_k:
                rho_k = tmp

    norm2_W1 = np.sqrt(np.sum((W1) ** 2))
    norm2_W2 = np.sqrt(np.sum((W2) ** 2))

    return rho_k/(norm2_W1*norm2_W2)

def avg_max_k_translation_correlation(k, W):
    '''
    Average maximum k-translation correlation

    :param k: maximum translation in x-axis and y-axis
    :param W: array like (nb_filters, depth, width, height), filters of a layer
    :return: Average maximum k-translation correlation on W, standard deviation on maximum k-translation correlation on W
    '''
    nb_filters, _, _, _ = W.shape
    avg_rho_k = 0

    rho_k_list = []
    for i in range(nb_filters):
        rho_k_max = 0
        for j in range(nb_filters):
            if i == j:
                continue
            rho_k = k_translation_correlation(k, W[i], W[j])
            rho_k_list.append(rho_k)
            if rho_k > rho_k_max:
                rho_k_max = rho_k
        avg_rho_k += rho_k_max

    return avg_rho_k/nb_filters, np.std(rho_k_list)


if __name__ == '__main__':
    import h5py
    # AlexNet weights : https://github.com/heuritech/convnets-keras
    # http://files.heuritech.com/weights/alexnet_weights.h5
    alexnet_weights = '../weights/alexnet_weights.h5'
    # get first layer filters learned by AlexNet
    with h5py.File(alexnet_weights, 'r') as f:
        W = np.array(f['conv_1']['conv_1_W'])

    print W[0].shape
    rho1 = k_translation_correlation(3, W[0], W[1])
    print rho1

    avg_rho_1, std_rho_1 = avg_max_k_translation_correlation(1,W)
    print avg_rho_1, std_rho_1

