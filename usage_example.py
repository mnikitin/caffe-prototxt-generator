#!/usr/bin/env python

import sys
from caffe_layers import *

def main(argc, argv):
    
    net_name = 'test.prototxt'
    with open(net_name, 'w') as f:
        set_data_deploy(f, 'data', 1, 3, 112, 112):
        set_data_lmdb(f, 'data', ['data', 'label'], '~/db/caffe/train_lmdb', 256, ['crop_size: 50', 'mirror: 1'], 'TRAIN')
        set_data_hdf5(f, 'input_clip', 'clip', '~/lstm_fr/hd5_clip.txt', 128, 'false', 'TRAIN')
        set_convolution(f, 'conv1_1', 'data', 'conv1_1', [['name: "conv1_1"' , 'lr_mult: 1', 'decay_mult: 0']], 64, 1, 3, 2, None, ['gaussian', 'std: 0.03'], 'constant')
        set_innerproduct(f, 'ip1', 'pool1', 'ip1', [], 100500, None, ['gaussian', 'std: 0.01'], ['constant', 'value: 0.0'])
        set_relu(f, 'relu1_1', 'conv1_1')
        set_pooling(f, 'pool1', 'conv1_1', 'pool1', 'MAX', 2, 2)
        set_dropout(f, 'drop1', 'pool1', 'pool1', 0.4)
        set_batchnorm(f, 'bn', 'ip1', 'ip1')
        set_scale(f, 'scale', 'ip1', 'ip1', [], 'true', None, ['constant', 'value: 0'])
        set_eltwise(f, 'eltwise_sum', ['ip1', 'ip2'], 'ip_sum', 'SUM')
        set_concat(f, 'concat1', ['conv1_1', 'conv1_2'], 'concat_conv', 1)
        set_loss_softmax(f, 'softmax_loss', ['ip1', 'label'], 'softmax_loss')
        set_loss_euclidean(f, 'euclidean_loss', ['ip1', 'ip1'], 'euclidean_loss', 0.95)


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)

