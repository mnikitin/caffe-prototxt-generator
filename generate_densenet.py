#!/usr/bin/env python

import sys
from caffe_layers import *

def set_bn_scale_relu_conv(fid, conv_name, bottom_name, num_output, pad, kernel_size, stride):
    set_batchnorm(fid, conv_name + '/bn', bottom_name, conv_name + '/bn')
    set_scale(fid, conv_name + '/scale', conv_name + '/bn', conv_name + '/bn', [], 'true', 1.0, 0.0)
    set_relu(fid, conv_name + '/relu', conv_name + '/bn', conv_name + '/bn')
    set_convolution(fid, conv_name, conv_name + '/bn', conv_name, [], num_output, pad, kernel_size, stride, 'false', 'msra')
    return conv_name

def set_bn_scale_relu_pool(fid, pool_name, bottom_name, pool_type, kernel_size, stride, global_pooling):
    set_batchnorm(fid, pool_name + '/bn', bottom_name, pool_name + '/bn')
    set_scale(fid, pool_name + '/scale', pool_name + '/bn', pool_name + '/bn', [], 'true', 1.0, 0.0)
    set_relu(fid, pool_name + '/relu', pool_name + '/bn', pool_name + '/bn')
    set_pooling(fid, pool_name, pool_name + '/bn', pool_name, pool_type, kernel_size, stride, global_pooling)
    return pool_name

def set_dense_block(fid, block_name, bottom_name, top_name, k, B):
    if B:
        set_bn_scale_relu_conv(fid, block_name + '/x1', bottom_name, 4*k, 0, 1, 1)
        set_bn_scale_relu_conv(fid, block_name + '/x2', block_name + '/x1', k, 1, 3, 1)
        set_concat(fid, top_name, [bottom_name, block_name + '/x2'], top_name)
    else:
        set_bn_scale_relu_conv(fid, block_name, bottom_name, k, 1, 3, 1)
        set_concat(fid, top_name, [bottom_name, block_name], top_name)

def set_dense_unit(fid, unit_id, bottom_name, num_dense_block, k, B):
    unit_id_str = str(unit_id)
    cur_bottom = bottom_name
    for i in range(1, num_dense_block+1):
        block_id = unit_id_str + '_' + str(i)
        block_name = 'conv' + block_id
        cur_top = 'concat' + block_id
        set_dense_block(fid, block_name, cur_bottom, cur_top, k, B)
        cur_bottom = cur_top
    return cur_bottom

def set_transition(fid, unit_id, bottom_name, C, num_input):
    unit_id_str = str(unit_id)
    conv_name = 'conv' + unit_id_str + '_blk'
    top_name = 'pool' + unit_id_str
    num_output = int(C*num_input)
    set_bn_scale_relu_conv(fid, conv_name, bottom_name, num_output, 0, 1, 1)
    set_pooling(fid, top_name, conv_name, top_name, 'AVE', 2, 2)
    return top_name, num_output


def set_densenet(fid, bottom_name, num_dense_block, k, B, C):
    set_convolution(fid, 'conv1', bottom_name, 'conv1', [], 2*k, 1, 3, 1, 'false', 'msra')   # Conv1
    top = set_bn_scale_relu_pool(fid, 'pool1', 'conv1', 'MAX', 2, 2, False)
    top = set_dense_unit(fid, 2, top, num_dense_block[0], k, B)                              # Conv2.x
    top, num = set_transition(fid, 2, top, C, 2*k + num_dense_block[0]*k)
    top = set_dense_unit(fid, 3, top, num_dense_block[1], k, B)                              # Conv3.x
    top, num = set_transition(fid, 3, top, C, num + num_dense_block[1]*k)
    top = set_dense_unit(fid, 4, top, num_dense_block[2], k, B)                              # Conv4.x
    top, num = set_transition(fid, 4, top, C, num + num_dense_block[2]*k)
    top = set_dense_unit(fid, 5, top, num_dense_block[3], k, B)                              # Conv5.x
    top = set_bn_scale_relu_pool(fid, 'pool5', top, 'AVE', 0, 0, True)                       # Embedding

def main(argc, argv):
    
    net_name = 'densenet.prototxt'
    k = 128
    B = True
    C = 0.5

    with open(net_name, 'w') as f:
        set_data_deploy(f, 'data', 1, 3, 112, 112)
        set_densenet(f, 'data', [2, 4, 6, 2], k, B, C)

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
