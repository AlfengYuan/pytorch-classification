"""
@author: Alfeng Yuan
"""
import torch
import traceback
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from graphviz import Digraph
from enum import Enum
import models
model_names = models.__all__
import sys

# Global Variable
INLINE = False # whether use origin pytorch api
dot = Digraph(name="netscope", comment="PyTorch", format="pdf")

LayerOut_id = dict({})# layerid to marke outputName
# registed nn layers
REGISTERED_LIST = [
    'Conv2d', 'ConvTranspose2d',
    'ConstantPad1d','ConstantPad2d', 'ConstantPad3d', 'ZeroPad2d',
    'Linear',
    'ReLU', 'LeakyReLU', 'Sigmoid', "Softmax", "ReLU6"
    'MaxPool2d', 'AvgPool2d', 
    'BatchNorm2d','BatchNorm1d', 'BatchNorm3d', 'InstanceNorm1d','InstanceNorm2d', 'InstanceNorm3d',
    'AdaptiveAvgPool2d'
    ]

class Blob_LOG():
    "Blob_LOG class for setting and getting data with id(data)"
    def __init__(self):
        self.data = {}
    
    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

class TransLog(object):
    "Core components for connecting up and down of network"
    def __init__(self):
        self.layers = {}
        self.detail_layers = {}
        self.detail_blobs = {}
        self._blobs = Blob_LOG()
        self._blobs_data = []
        
    def init(self, inputs, name=None):
        "init input data"
        name = "data" if name is None else name
        LayerOut_id[int(id(inputs))] = name
        self.add_blobs(inputs, name=name )
        dot.node(name=name, label=name, style='filled', fillcolor="green")

    def add_layer(self, name='layer'):
        if name in self.layers:
            return self.layers[name]
        if name not in self.detail_layers.keys():
            self.detail_layers[name] = 0
        self.detail_layers[name] += 1
        name = '{}{}'.format(name, self.detail_layers[name])
        self.layers[name] = name
        #print("{} was added to layers".format(self.layers[name]))
        return self.layers[name]

    def add_blobs(self, blobs, name='blob'):
        rst = []
        for blob in blobs:
            self._blobs_data.append(blob)
            blob_id = int(id(blob))
            rst.append(name)
            #print("{}: {} was added to blobs".format(blob_id, rst[-1]))
            self._blobs[blob_id] = rst[-1]
        return rst

    def blobs(self, var):
        var = id(var)
        try:
            return self._blobs[var]
        except:
            print("WARRING: CANNOT FOUND blob {}".format(var))
            return None
log = TransLog()

# Refactor each op 
# add layer's params to json data and generating wgt

# nn.Conv2d-->F.conv2d
def _conv2d(raw, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    x = raw(input, weight, bias, stride, padding, dilation, groups)
    INLINE = True
    name = log.add_layer(name="conv2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    node_label = name + "\n" + "weight: {}".format(str(list(weight.size())))
    if bias is not None:
        node_label += "\n" + "bias: {}".format(str(list(bias.size())))
    if stride[0] > 1 or stride[1] > 1:
        node_label += "\n" + "stride: {}".format(stride)
    if padding[0] > 0 or padding[1] > 0:
        node_label += "\n" + "padding: {}".format(padding)
    if dilation[0] > 1 or dilation[1] > 1:
        node_label += "\n" + "dilation: {}".format(dilation)
    if groups > 1:
        node_label += "\n" + "groups: {}".format(groups)
    dot.node(name=name, label=node_label)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]----> F.batch_norm----> torch.batch_norm
def _batch_norm(raw, input, weight, bias, running_mean, running_var, training, momentum, eps, torch_backends_cudnn_enabled):
    x = raw(input, weight, bias, running_mean, running_var, training, momentum, eps, torch_backends_cudnn_enabled)
    INLINE = True
    name = log.add_layer(name="BN_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# [nn.ReLU6--->nn.Hardtanh--->F.hardtanh]
def _hardtanh(raw, input, min_val=-1., max_val=1., inplace=False):
    x = raw(input, min_val, max_val, inplace)
    INLINE = True
    name = log.add_layer(name="hardtanh_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# nn.ReLU----->F.relu
def _relu(raw, input, inplace=False):
    x = raw(input, inplace)
    INLINE = True
    name = log.add_layer(name="relu_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    
    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# nn.LeakyReLU---->F.leaky_relu
def _leaky_relu(raw, input, negative_slope=0.01, inplace=False):
    x = raw(input, negative_slope, inplace)
    INLINE = True
    name = log.add_layer(name="leaky_relu_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# nn.Sigmoid----->torch.sigmoid
def _sigmoid(raw, input):
    x = raw(input)
    INLINE = True
    name = log.add_layer(name="sigmoid_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name)
    INLINE = False
    return x

# torch.clamp
def _clamp(raw, input, min, max):
    x = raw(input, min, max)
    INLINE = True
    name = log.add_layer(name="clamp_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name)
    INLINE = False
    return x

# nn.MaxPool2d----->F.max_pool2d
def _max_pool2d(raw, *args, **kwargs):
    # args = (input, kernel, stride, padding, dilation, ceil_mode, return_indices)
    x = raw(*args, **kwargs)
    INLINE = True
    name = log.add_layer(name="max_pool2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    
    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(args[0]), name, label=str(list(args[0].size())))
    INLINE = False
    return x

# nn.AvgPool2d------>F.avg_pool2d
def _avg_pool2d(raw, input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    x = raw(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    INLINE = True
    name = log.add_layer(name="avg_pool2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# nn.Linear---->F.linear
def _linear(raw, input, weight, bias=None):
    x = raw(input, weight, bias)
    INLINE = True
    name = log.add_layer(name="linear_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    node_label = name + "\n" + "weight: {}".format(str(list(weight.size())))
    if bias is not None:
        node_label += "\n" + "bias: {}".format(str(list(bias.size())))
    dot.node(name=name, label=node_label)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# torch.flatten
def _flatten(raw, input, start_dim=0, end_dim=-1):
    x = raw(input, start_dim, end_dim)
    INLINE = True
    name = log.add_layer(name="flatten_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# torch.cat
def _cat(raw, inputs, dim=0):
    x = raw(inputs, dim)
    INLINE = True
    name = log.add_layer(name="cat_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    for i in range(len(inputs)):
        dot.edge(log.blobs(inputs[i]), name, label=str(list(inputs[i].size())))
    INLINE = False
    return x

# nn.Softmax--->F.softmax
def _softmax(raw, input, dim=None, _stacklevel=3, dtype=None):
    x = raw(input, dim, _stacklevel, dtype)
    INLINE = True
    name = log.add_layer(name="softmax_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name)
    INLINE = False
    return x

# torch.mean
def _mean(raw, input, dim, keepdim=False):
    x = raw(input, dim, keepdim)
    INLINE = True
    name = log.add_layer(name="mean_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x


# ['InstanceNorm1d','InstanceNorm2d', 'InstanceNorm3d']------>F.instance_norm---->torch.instance_norm
def _instance_norm(raw, input, weight, bias, running_mean, running_var,  use_input_stats, momentum, eps, torch_backends_cudnn_enabled):
    x = raw(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, torch_backends_cudnn_enabled)
    INLINE = True
    name = log.add_layer(name="IN_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# ConvTranspose2d---->F.conv_transpose2d
def _conv_transpose2d(raw, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x = raw(input, weight, bias, stride, padding, output_padding, groups, dilation)
    INLINE = True
    name = log.add_layer(name="Deconv2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    node_label = name + "\n" + "weight: {}".format(str(list(weight.size())))
    if bias is not None:
        node_label += "\n" + "bias: {}".format(str(list(bias.size())))
    dot.node(name=name, label=node_label)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x
    
# ['ConstantPad1d','ConstantPad2d', 'ConstantPad3d', 'ZeroPad2d']----->F.pad
def _pad(raw, input, pad, mode="constant", value=0):
    x = raw(input, pad, mode, value)
    INLINE = True
    name = log.add_layer(name="pad_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# torch.topk
def _topk(raw, input, k, dim=None, largest=True, sorted=True):
    x = raw(input, k, dim, largest, sorted)
    INLINE = True
    name = log.add_layer(name="topk_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# torch.argmax
def _argmax(raw, input, dim, keepdim=False):
    x = raw(input, dim, keepdim)
    INLINE = True
    name = log.add_layer(name="argmax_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# F.interpolate ,  here we use Down/up samples the input to either the given size (resize, upsampling, downsampling)
def _interpolate(raw, input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    x = raw(input, size, scale_factor, mode, align_corners, recompute_scale_factor)
    INLINE = True
    name = log.add_layer(name="interpolate_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# F.relu6
def _relu6(raw, input, inplace=False):
    x = F.hardtanh(input, 0., 6., inplace)
    return x


# unaryop
def _unaryop(raw, style, input):
    x = raw(input)
    INLINE = True
    name = log.add_layer(name="unaryop_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
   
    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name)
    INLINE = False
    return x

# _add
def _add(input, *args):
    x = raw__add__(input, *args)
    name = log.add_layer(name="add_")
    log.add_blobs([x], name=name)
    if log.blobs(args[0]) is None:
        const_name = log.add_layer(name="add_const_")
        log.add_blobs([args[0]], name=const_name)
    LayerOut_id[int(id(x))] = name

    # netscope
    dot.node(name=name, label=name)
    for n in [log.blobs(input), log.blobs(args[0])]:
        dot.edge(n, name)
    return x

# _sub
def _sub(input, *args):
    x = raw__sub__(input, *args)
    name = log.add_layer(name="sub_")
    log.add_blobs([x], name=name)
    if log.blobs(args[0]) is None:
        const_name = log.add_layer(name="sub_const_")
        log.add_blobs([args[0]], name=const_name)
    LayerOut_id[int(id(x))] = name
   
    # netscope
    dot.node(name=name, label=name)
    for n in [log.blobs(input), log.blobs(args[0])]:
        dot.edge(n, name)
    return x

# _expand_as
def _expand_as(input, *args):
    x = raw__expand_as__(input, *args)
    name = log.add_layer(name="expand_as_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    
    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    return x

# _view
def _view(input, *args):
    x1 = raw__view__(input, *args)
    x = torch.reshape(input, x1.shape)
    return x

# _permute
def _permute(input, *args):
    x = raw__permute__(input, *args)
    name = log.add_layer(name="permute_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
 
    # netscope
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    return x

# torch.div
def _div(raw, input, other):
    x = raw(input, other)
    INLINE = True
    name = log.add_layer(name="div_")
    log.add_blobs([x], name=name)
    if log.blobs(other) is None:
        const_name = log.add_layer(name="div_const_")
        log.add_blobs([other], name=const_name)
    LayerOut_id[int(id(x))] = name
 
    # netscope
    dot.node(name=name, label=name)
    for n in [log.blobs(input), log.blobs(other)]:
        dot.edge(n, name)
    INLINE = False
    return x

# torch.split
def _split(raw, tensor, split_size_or_sections, dim=0):
    x = raw(tensor, split_size_or_sections, dim)
    INLINE = True
    name = log.add_layer(name="split_")
    layerName = []
    start = 0
    slicePoint = [start, ]
    for i in range(len(x)):
        layerName.append(name+"_idx{}".format(i+1))
        log.add_blobs([x[i]], name=layerName[-1])
        LayerOut_id[int(id(x[i]))] = layerName[-1]
        start += len(x[i])
        slicePoint.append(start)
    
    # netscope
    for n in layerName:
        dot.node(name=n, label=n)
        dot.edge(log.blobs(tensor), n, label=str(list(tensor.size())))
    INLINE = False
    return x

# torch.chunk
def _chunk(raw, input, chunks, dim=0):
    x = raw(input, chunks, dim)
    INLINE = True
    name = log.add_layer(name="chunk_")
    layerName = []
    for i in range(len(x)):
        layerName.append(name + "_idx{}".format(i+1))
        log.add_blobs([x[i]], name=layerName[-1])
        LayerOut_id[int(id(x[i]))] = layerName[-1]

    # netscope
    for n in layerName:
        dot.node(name=n, label=n)
        dot.edge(log.blobs(input), n, label=str(list(input.size())))
    INLINE = False
    return x

 # torch.reshape   
def _reshape(raw, input, shape):
    x = raw(input, shape)
    INLINE = True
    name = log.add_layer(name="reshape_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
   
    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# torch.index_select
def _index_select(raw, input, dim, index):
    x = raw(input, dim, index)
    INLINE = True
    name = log.add_layer(name="index_select_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# torch.unsqueeze
def _unsqueeze(raw, input, dim):
    x = raw(input, dim)
    INLINE = True
    name = log.add_layer(name="unsqueeze_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# torch.bmm
def _bmm(raw, input, mat2):
    x = _matmul(raw, input, mat2)
    return x
    
# torch.matmul
def _matmul(raw, input, other):
    x = raw(input, other)
    INLINE = True
    name = log.add_layer(name="matmul_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    
    dot.node(name=name, label=name)
    for n in [input, other]:
        dot.edge(log.blobs(n), name, label=str(list(n.size())))
    INLINE = False
    return x

# torch.sum
def _sum(raw, input):
    x = raw(input)
    INLINE = True
    name = log.add_layer(name="sum_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name)
    INLINE = False
    return x

def _sum(raw, input, dim, keepdim=False):
    x = raw(input, dim, keepdim)
    INLINE = True
    name = log.add_layer(name="sum_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# torch.transpose
def _transpose(raw, input, dim0, dim1):
    x = raw(input, dim0, dim1)
    INLINE = True
    name = log.add_layer(name="transpose_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x


# nn.AdaptiveAvgPool2d----->F.adaptive_avg_pool2d
def _adaptive_avg_pool2d(raw, input, output_size):
    x = raw(input, output_size)
    INLINE = True
    name = log.add_layer(name="adaptive_avg_pool2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    dot.node(name=name, label=name)
    dot.edge(log.blobs(input), name, label=str(list(input.size())))
    INLINE = False
    return x

# a = a * b
def _mul(input, *args):
    x = raw__mul__(input, *args)
    name = log.add_layer(name="mul_")
    log.add_blobs([x], name=name)
    if log.blobs(args[0]) is None:
        const_name = log.add_layer(name="mul_const_")
        log.add_blobs([args[0]], name=const_name)
    LayerOut_id[int(id(x))] = name
 
    dot.node(name=name, label=name)
    for n in [log.blobs(input), log.blobs(args[0])]:
        dot.edge(n, name)
    return x

# t.contiguous
def _contiguous(input, *args):
    x = raw__contiguous__(input, *args)
    if int(id(x)) != int(id(input)):
        name = log.add_layer(name="contiguous_")
        log.add_blobs([x], name=name)
        LayerOut_id[int(id(x))] = name
        dot.node(name=name, label=name)
        dot.edge(log.blobs(input), name, label=str(list(input.size())))
    return x


class Rp(object):
    def __init__(self, raw, replace, **kwargs):
        # replace the raw function to replace funtion
        self.obj = replace
        self.raw = raw

    def __call__(self, *args, **kwargs):
        if INLINE:
            return self.raw(*args, **kwargs)
        else:
            out = self.obj(self.raw, *args, **kwargs)
            return out
    
UnaryOperationStyle = Enum(("UnaryOperationStyle"), 
                           ("kEXP", "kLOG", "kSQRT", "kRECIP", "kABS", "kNEG", "kSIN", "kCOS", "kTAN", "kSINH", "kCOSH", 
                            "kASIN", "kACOS", "kATAN", "kASINH", "kACOSH", "kATANH", "kCEIL", "kFLOOR", "kERF", "kNOT"), start=0)
class UnaryOperation(object):
    def __init__(self, raw, replace, style, **kwargs):
        # replace the raw function to replace funtion
        self.obj = replace
        self.raw = raw
        self.style = style

    def __call__(self, *args, **kwargs):
        if INLINE:
            return self.raw(*args, **kwargs)
        else:
            out = self.obj(self.raw, self.style, *args, **kwargs)
            return out



# Registration list about all support op
F.conv2d = Rp(F.conv2d, _conv2d)
F.relu = Rp(F.relu, _relu)
F.leaky_relu = Rp(F.leaky_relu, _leaky_relu)
F.max_pool2d = Rp(F.max_pool2d, _max_pool2d)
F.avg_pool2d = Rp(F.avg_pool2d, _avg_pool2d)
F.linear = Rp(F.linear, _linear)
F.adaptive_avg_pool2d = Rp(F.adaptive_avg_pool2d, _adaptive_avg_pool2d)
F.softmax = Rp(F.softmax, _softmax)
F.conv_transpose2d = Rp(F.conv_transpose2d, _conv_transpose2d)
F.pad = Rp(F.pad, _pad)
F.interpolate = Rp(F.interpolate, _interpolate)
F.hardtanh = Rp(F.hardtanh, _hardtanh)
F.relu6 = Rp(F.relu6, _relu6)
# torch op
torch.clamp = Rp(torch.clamp, _clamp)
torch.batch_norm = Rp(torch.batch_norm, _batch_norm)
torch.sigmoid = Rp(torch.sigmoid, _sigmoid)
torch.flatten = Rp(torch.flatten, _flatten)
torch.cat = Rp(torch.cat, _cat)
torch.instance_norm = Rp(torch.instance_norm, _instance_norm)
torch.topk = Rp(torch.topk, _topk)
torch.argmax = Rp(torch.argmax, _argmax)
torch.matmul = Rp(torch.matmul, _matmul)
torch.div = Rp(torch.div, _div) 
torch.split = Rp(torch.split, _split) 
torch.chunk = Rp(torch.chunk, _chunk)
torch.reshape = Rp(torch.reshape, _reshape) 
torch.unsqueeze = Rp(torch.unsqueeze, _unsqueeze)
torch.index_select = Rp(torch.index_select, _index_select)
torch.transpose = Rp(torch.transpose, _transpose)
torch.mean = Rp(torch.mean, _mean)
torch.bmm = Rp(torch.bmm, _bmm)
torch.sum = Rp(torch.sum, _sum)

# unary op
torch.exp = UnaryOperation(torch.exp, _unaryop, UnaryOperationStyle.kEXP)
torch.log = UnaryOperation(torch.log, _unaryop, UnaryOperationStyle.kLOG)
torch.sqrt = UnaryOperation(torch.sqrt, _unaryop, UnaryOperationStyle.kSQRT)
torch.reciprocal = UnaryOperation(torch.reciprocal, _unaryop, UnaryOperationStyle.kRECIP)
torch.abs = UnaryOperation(torch.abs, _unaryop, UnaryOperationStyle.kABS)
torch.neg = UnaryOperation(torch.neg, _unaryop, UnaryOperationStyle.kNEG)
torch.sin = UnaryOperation(torch.sin, _unaryop, UnaryOperationStyle.kSIN)
torch.cos = UnaryOperation(torch.cos, _unaryop, UnaryOperationStyle.kCOS)
torch.tan = UnaryOperation(torch.tan, _unaryop, UnaryOperationStyle.kTAN)
torch.sinh = UnaryOperation(torch.sinh, _unaryop, UnaryOperationStyle.kSINH)
torch.cosh = UnaryOperation(torch.cosh, _unaryop, UnaryOperationStyle.kCOSH)
torch.asin = UnaryOperation(torch.asin, _unaryop, UnaryOperationStyle.kASIN)
torch.acos = UnaryOperation(torch.acos, _unaryop, UnaryOperationStyle.kACOS)
torch.atan = UnaryOperation(torch.atan, _unaryop, UnaryOperationStyle.kATAN)
torch.asinh = UnaryOperation(torch.asinh, _unaryop, UnaryOperationStyle.kASINH)
torch.acosh = UnaryOperation(torch.acosh, _unaryop, UnaryOperationStyle.kACOSH)
torch.atanh = UnaryOperation(torch.atanh, _unaryop, UnaryOperationStyle.kATANH)
torch.ceil = UnaryOperation(torch.ceil, _unaryop, UnaryOperationStyle.kCEIL)
torch.floor = UnaryOperation(torch.floor, _unaryop, UnaryOperationStyle.kFLOOR)
torch.erf = UnaryOperation(torch.erf, _unaryop, UnaryOperationStyle.kERF)
torch.logical_not = UnaryOperation(torch.logical_not, _unaryop, UnaryOperationStyle.kNOT)
# Tensor op
for t in [torch.Tensor]:
    # c = a + b
    raw__add__ = t.__add__ 
    t.__add__ = _add
    # c = a - b
    raw__sub__ = t.__sub__
    t.__sub__ = _sub
    # c = a * b 
    raw__mul__ = t.__mul__
    t.__mul__ = _mul

    raw__view__ = t.view
    t.view = _view
    raw__permute__ = t.permute
    t.permute = _permute
    raw__expand_as__ = t.expand_as
    t.expand_as = _expand_as
    raw__contiguous__ = t.contiguous
    t.contiguous = _contiguous

    


# main function to run
def run(net, input_var):
    print("Starting........")
    INLINE = False
    net.eval()
    log.init([input_var]) 
    with torch.no_grad():
        outs = net(input_var)
    INLINE = True
    # mark output layer
    # TODO: change too more flexible, now onlu support batchsize=1, it's a known bug
    if len(outs) >=2:
        for i, out in enumerate(outs):
            dot.node(LayerOut_id[int(id(out))], style="filled", fillcolor='red')
    elif len(outs)==1:
        dot.node(LayerOut_id[int(id(outs))], style="filled",  fillcolor='red')

    dot.render(filename=sys.argv[1], directory="./",view=False)
    print()
    print("Successed !")
    return outs


if __name__ == "__main__":
    if sys.argv[1] in ["googlenet", "inception_v3"]:
        net = model_names[sys.argv[1]](pretrained=True, transform_input=False)
    else:
        net = model_names[sys.argv[1]](pretrained=False)
    x = torch.ones([1, 3, 224, 224])
    run(net, x)