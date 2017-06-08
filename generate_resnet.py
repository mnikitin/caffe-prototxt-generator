#!/usr/bin/env python

import sys
from caffe_layers import *

def set_conv_bn_scale_relu(fid, block_id, bottom_name, num_output, pad, kernel_size, stride):
    name = 'conv' + block_id
    set_convolution(fid, name, bottom_name, name, [], num_output, pad, kernel_size, stride, None, 'msra')
    set_batchnorm(fid, name + '_bn', name, name)
    set_scale(fid, name + '_scale', name, name, [], 'true')
    set_relu(fid, name + '_relu', name, name)

def set_residual_block(fid, block_id, bottom_name, top_name, num_conv, num_output, pad, kernel_size, stride):
    cur_bottom = bottom_name
    for i in range(1, num_conv+1):
        cur_id = block_id + '_' + str(i)
        set_conv_bn_scale_relu(fid, cur_id, cur_bottom, num_output, pad, kernel_size, stride)
        cur_bottom = 'conv' + cur_id
    top_name = 'conv' + block_id + '_sum'
    set_eltwise(fid, top_name, ['conv' + cur_id, bottom_name], top_name, 'SUM')

def set_conv_unit(fid, unit_id, bottom_name, num_res_block, num_res_conv, num_output):
    unit_id_str = str(unit_id)
    set_conv_bn_scale_relu(fid, unit_id_str + '_1', bottom_name, num_output, 1, 3, 2)
    cur_bottom = 'conv' + unit_id_str + '_1'
    for i in range(2, num_res_block+2):
        res_block_id = unit_id_str + '_' + str(i)
        cur_top = 'conv' + res_block_id + '_sum'
        set_residual_block(fid, res_block_id, cur_bottom, cur_top, num_res_conv, num_output, 1, 3, 1)
        cur_bottom = cur_top
    return cur_bottom

def set_resnet(fid, bottom_name, num_res_block):
    top = set_conv_unit(fid, 1, bottom_name, num_res_block[0], 2, 64)       # Conv1.x
    top = set_conv_unit(fid, 2, top, num_res_block[1], 2, 128)              # Conv2.x
    top = set_conv_unit(fid, 3, top, num_res_block[2], 2, 256)              # Conv3.x
    top = set_conv_unit(fid, 4, top, num_res_block[3], 2, 512)              # Conv4.x
    set_innerproduct(fid, 'fc5', top, 'fc5', [], 512, None, 'msra')           # FC5

def main(argc, argv):
    
    net_name = 'resnet20.prototxt'
    with open(net_name, 'w') as f:
        set_data_deploy(f, 'data', 1, 3, 112, 112)
        set_resnet(f, 'data', [1, 2, 4, 1])


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
