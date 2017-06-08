#!/usr/bin/env python

from caffe_layers import *

def main(argc, argv):
    
    net_name = 'test.prototxt'
    with open(net_name, 'w') as f:
        set_convolution(f, 'conv1_1', 'data', 'conv1_1', [['name: "conv1_1"' , 'lr_mult: 1', 'decay_mult: 0']], 64, 1, 3, 2, None, ['gaussian', 'std: 0.03'], 'constant')
        set_relu(f, 'relu1_1', 'conv1_1')
        set_pooling(f, 'pool1', 'conv1_1', 'pool1', 'MAX', 2, 2)
        set_dropout(f, 'drop1', 'pool1', 'pool1', 0.4)
        set_innerproduct(f, 'ip1', 'pool1', 'ip1', [], 100500, None, ['gaussian', 'std: 0.01'], ['constant', 'value: 0.0'])
        

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
