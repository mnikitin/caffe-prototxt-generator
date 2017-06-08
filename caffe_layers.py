#!/usr/bin/env python

def create_layer_head(fid, layer_name, layer_type, bottom_name, top_name):
    if top_name is None:
        top_name = bottom_name  # in-place
    fid.write("layer {\n")
    fid.write("  name: \"%s\"\n" % (layer_name))
    fid.write("  type: \"%s\"\n" % (layer_type))
    # bottom
    if type(bottom_name) is str:
        fid.write("  bottom: \"%s\"\n" % (bottom_name))
    elif type(bottom_name) is list:
        for bot in bottom_name:
         fid.write("  bottom: \"%s\"\n" % (bot))
    # top
    if type(top_name) is str:
        fid.write("  top: \"%s\"\n" % (top_name))
    elif type(top_name) is list:
        for top in top_name:
            fid.write("  top: \"%s\"\n" % (top))

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


def set_data_lmdb(fid, layer_name, top_name, source, batch_size, transform_param, include_phase):
    create_layer_head(fid, layer_name, 'Data', [], top_name)
    fid.write("  data_param {\n")
    fid.write("    source: \"%s\"\n" % (source))
    fid.write("    backend: LMDB\n")
    fid.write("    batch_size: %d\n" % (batch_size))
    fid.write("  }\n")
    if transform_param:
        fid.write("  transform_param {\n")
        for elem in transform_param:
            fid.write("    %s\n" % (elem))
        fid.write("  }\n")
    fid.write("  include: { phase: %s }\n" % (include_phase))
    fid.write("}\n")

def set_data_hdf5(fid, layer_name, top_name, source, batch_size, shuffle, include_phase):
    create_layer_head(fid, layer_name, 'HDF5Data', [], top_name)
    fid.write("  hdf5_data_param {\n")
    fid.write("    source: \"%s\"\n" % (source))
    fid.write("    batch_size: %d\n" % (batch_size))
    fid.write("    shuffle: %s\n" % (shuffle))
    fid.write("  }\n")
    fid.write("  include: { phase: %s }\n" % (include_phase))
    fid.write("}\n")

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

def set_batchnorm(fid, layer_name, bottom_name, top_name):
    create_layer_head(fid, layer_name, 'BatchNorm', bottom_name, top_name)
    for _ in range(3):
        set_param(fid, ['lr_mult: 0', 'decay_mult: 0'])
    fid.write("}\n")

def set_scale(fid, layer_name, bottom_name, top_name, params, bias_term, filler, bias_filler):
    create_layer_head(fid, layer_name, 'Scale', bottom_name, top_name)
    for param in params:
        set_param(fid, param)
    if bias_term is not None or filler is not None or bias_filler is not None:
        fid.write("  scale_param {\n")
        fid.write("    bias_term: %s\n" % (bias_term))
        set_filler(fid, 'filler', filler)
        set_filler(fid, 'bias_filler', bias_filler)
        fid.write("  }\n")
    fid.write("}\n")

def set_eltwise(fid, layer_name, bottom_name, top_name, operation):
    create_layer_head(fid, layer_name, 'Eltwise', bottom_name, top_name)
    fid.write("  eltwise_param {\n")
    fid.write("    operation: %s\n" % (operation))
    fid.write("  }\n")
    fid.write("}\n")


