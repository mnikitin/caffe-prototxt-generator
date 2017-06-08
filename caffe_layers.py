#!/usr/bin/env python

def create_layer_head(fid, layer_name, layer_type, bottom_name, top_name):
    if top_name is None:
        top_name = bottom_name  # in-place
    fid.write("layer {\n")
    fid.write("  name: \"%s\"\n" % (layer_name))
    fid.write("  type: \"%s\"\n" % (layer_type))
    fid.write("  bottom: \"%s\"\n" % (bottom_name))
    fid.write("  top: \"%s\"\n" % (top_name))

def set_filler(fid, filler_name, filler):
    if filler:
        fid.write("    %s {\n" % (filler_name))
        if type(filler) is str:
            fid.write("      type: \"%s\"\n" % (filler))
        elif type(filler) is list:
            fid.write("      type: \"%s\"\n" % (filler[0]))
            for elem in filler[1:]:
                fid.write("      %s\n" % (elem))
        fid.write("    }\n")

def set_param(fid, param):
    fid.write("  param {\n")
    for elem in param:
        fid.write("    %s\n" % (elem))
    fid.write("  }\n")


def set_convolution(fid, layer_name, bottom_name, top_name, params, num_output, pad, kernel_size, stride, bias_term = None, weight_filler = None, bias_filler = None):
    create_layer_head(fid, layer_name, 'Convolution', bottom_name, top_name)
    for param in params:
        set_param(fid, param)
    fid.write("  convolution_param {\n")
    fid.write("    num_output: %d\n" % (num_output))
    fid.write("    pad: %d\n" % (pad))
    fid.write("    kernel_size: %d\n" % (kernel_size))
    fid.write("    stride: %d\n" % (stride))
    if bias_term is not None:
        fid.write("    bias_term: %s\n" % (bias_term))
    set_filler(fid, 'weight_filler', weight_filler)
    set_filler(fid, 'bias_filler', bias_filler)
    fid.write("  }\n")
    fid.write("}\n")

def set_innerproduct(fid, layer_name, bottom_name, top_name, params, num_output, bias_term = None, weight_filler = None, bias_filler = None):
    create_layer_head(fid, layer_name, 'InnerProduct', bottom_name, top_name)
    for param in params:
        set_param(fid, param)
    fid.write("  inner_product_param {\n")
    fid.write("    num_output: %d\n" % (num_output))
    if bias_term is not None:
        fid.write("    bias_term: %s\n" % (bias_term))
    set_filler(fid, 'weight_filler', weight_filler)
    set_filler(fid, 'bias_filler', bias_filler)
    fid.write("  }\n")
    fid.write("}\n")

def set_relu(fid, layer_name, bottom_name, top_name = None):
    create_layer_head(fid, layer_name, 'ReLU', bottom_name, top_name)
    fid.write("}\n")

def set_pooling(fid, layer_name, bottom_name, top_name, pool_type, kernel_size, stride):
    create_layer_head(fid, layer_name, 'Pooling', bottom_name, top_name)
    fid.write("  pooling_param {\n")
    fid.write("    pool: %s\n" % (pool_type))
    fid.write("    kernel_size: %d\n" % (kernel_size))
    fid.write("    stride: %d\n" % (stride))
    fid.write("  }\n")
    fid.write("}\n")

def set_dropout(fid, layer_name, bottom_name, top_name, dropout_ratio):
    create_layer_head(fid, layer_name, 'Dropout', bottom_name, top_name)
    fid.write("  dropout_param {\n")
    fid.write("    dropout_ratio: %f\n" % (dropout_ratio))
    fid.write("  }\n")
    fid.write("}\n")

