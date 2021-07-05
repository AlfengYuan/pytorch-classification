"""
@author: AlfengYuan, Shanghai
@data: 2020-12-24 Merry Christmas!
"""
import torch
import traceback
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from Auto_Json import config as cfg
from .get_wgt import get_wgt
from enum import Enum

# Global Variable
DEBUG = cfg.DEBUG
INLINE = False # whether use origin pytorch api

LayerOut_id = dict({})# layerid to marke outputName
# registed nn layers
REGISTERED_LIST = [
    ###################TensorRT Support############################
    'Conv2d', 'ConvTranspose2d',
    'ConstantPad1d','ConstantPad2d', 'ConstantPad3d', 'ZeroPad2d',
    'Linear',
    'ReLU', 'LeakyReLU', 'Sigmoid', "Softmax",
    'MaxPool2d', 'AvgPool2d', 
    'BatchNorm2d','BatchNorm1d', 'BatchNorm3d', 'InstanceNorm1d','InstanceNorm2d', 'InstanceNorm3d',
    #################TensorRT not support class###############
    'AdaptiveAvgPool2d'
    ]

class Json():
    "Json class to dump json_file"
    def __init__(self, json_file):
        self.json_file = json_file
        self.data = {}
        self.data['input_c'] = cfg.INPUT_C
        self.data['input_h'] = cfg.INPUT_H
        self.data['input_w'] = cfg.INPUT_W
        self.data['ENGPath'] = cfg.ENG_PATH
        self.data['onnxPath'] = cfg.ONNXPATH
        self.data['weightsDir'] = cfg.WEIGHTS_DIR
        if not os.path.exists(self.data['weightsDir']):
            os.makedirs(self.data['weightsDir'])
        if cfg.INT8:
            self.data['int8'] = cfg.INT8
        else:
            self.data['fp16'] = cfg.FP16
        
        self.data['cali_txt'] = cfg.CALI_TXT
        self.data['cali_table'] = cfg.CALI_TABLE
        self.data['Mean'] = cfg.MEAN
        self.data['Std'] =  cfg.STD
        self.data['inputBlobName'] = cfg.INPUTBLOBNAME
        self.data['outputBlobName'] = cfg.OUTPUTBLOBNAME
        self.data['maxBatchsize'] = cfg.MAXBATCHSIZE
        self.data['outputSize'] = cfg.OUTPUTSIZE
        self.data['network'] = []
    
    def dump(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
            
js = Json(cfg.JSON_FILE_NAME)

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
        
    def init(self, inputs):
        "init input data"
        LayerOut_id[int(id(inputs))] = "data"
        self.add_blobs(inputs, name="data")

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

    # extract weight to get wgt
    get_wgt(weight, f'{name}.weight')
    weightFile = f'{name}'
    biasFile = f'{name}'
    # add json params
    if bias is not None:
        get_wgt(bias, f'{name}.bias')
        #biasFile = f'{name}'
    conv_params = dict({"layerStyle": "conv",
     "layerName": name,
     "inputName": log.blobs(input),
     "weightFile": weightFile,
     "biasFile": biasFile,
     "parameter": {
         "input_c": input.shape[1],
         "output_c": x.shape[1],
         "kernel": [weight.shape[2], weight.shape[3]],
         "padding": padding,
         "stride": stride,
         "dilation": dilation,
         "groups": groups
         }
     })
    #if bias is not None:
    #conv_params["biasFile"] = biasFile
    if DEBUG:
        print(conv_params)
    js.data['network'].append(conv_params)
    INLINE = False
    return x

# [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]----> F.batch_norm----> torch.batch_norm
def _batch_norm(raw, input, weight, bias, running_mean, running_var, training, momentum, eps, torch_backends_cudnn_enabled):
    x = raw(input, weight, bias, running_mean, running_var, training, momentum, eps, torch_backends_cudnn_enabled)
    INLINE = True
    name = log.add_layer(name="BN_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # extract weight to get wgt
    get_wgt(weight, f'{name}.weight')
    get_wgt(bias, f'{name}.bias')
    get_wgt(running_mean, f'{name}.running_mean')
    get_wgt(running_var, f'{name}.running_var')
    
    # add json params
    #weightFile = [f'{name}.weight', f'{name}.bias', f'{name}.running_mean', f'{name}.running_var']
    weightFile = f'{name}'
    bn_params = dict({"layerStyle": "bn",
     "layerName": name,
     "inputName": log.blobs(input),
     "weightFile": weightFile
    })
    if DEBUG:
        print(bn_params)
    js.data['network'].append(bn_params)
    INLINE = False
    return x

# nn.ReLU----->F.relu
def _relu(raw, input, inplace=False):
    x = raw(input, inplace)
    INLINE = True
    name = log.add_layer(name="relu_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    
    # no weight to extract
    # add json params
    relu_params = dict({"layerStyle": "active",
                        "layerName": name,
                        "inputName": log.blobs(input),
                        "active_type": "relu"#"kRELU"
                        })
    if DEBUG:
        print(relu_params)
    js.data['network'].append(relu_params)
    INLINE = False
    return x

# nn.LeakyReLU---->F.leaky_relu
def _leaky_relu(raw, input, negative_slope=0.01, inplace=False):
    x = raw(input, negative_slope, inplace)
    INLINE = True
    name = log.add_layer(name="leaky_relu_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight to extract
    # add json params
    leaky_relu_params = dict({"layerStyle": "active",
                              "layerName": name,
                              "inputName": log.blobs(input),
                              "active_type": "l_relu"})
    if DEBUG:
        print(leaky_relu_params)
    js.data['network'].append(leaky_relu_params)
    INLINE = False
    return x

# nn.Sigmoid----->torch.sigmoid
def _sigmoid(raw, input):
    x = raw(input)
    INLINE = True
    name = log.add_layer(name="sigmoid_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    #no weight to extract
    # add json params
    sigmoid_params = dict({"layerStyle": "active",
                           "layerName": name,
                           "inputName": log.blobs(input),
                           "active_type": "sigmoid"})
    if DEBUG:
        print(sigmoid_params)
    js.data['network'].append(sigmoid_params)
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
    
    #no weight to extract
    #add json params
    max_pool2d_params = dict({"layerStyle": "pool",
                              "layerName": name,
                              "inputName": log.blobs(args[0]),
                              "parameter":{
                                  "poolType": "kMAX",
                                  "kernel": [args[1], args[1]] if isinstance(args[1], int) else args[1] ,
                                  "stride": [args[2], args[2]] if isinstance(args[2], int) else args[2],
                                  "padding": [args[3], args[3]] if isinstance(args[3], int) else args[3]
                              }
                            })
    if DEBUG:
        print(max_pool2d_params)
    js.data['network'].append(max_pool2d_params)
    INLINE = False
    return x

# nn.AvgPool2d------>F.avg_pool2d
def _avg_pool2d(raw, input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    x = raw(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
    INLINE = True
    name = log.add_layer(name="avg_pool2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight to extract
    # add json params
    avg_pool2d_params = dict({
        "layerStyle": "pool",
        "layerName": name,
        "inputName": log.blobs(input),
        "parameter":{
            "poolType": "kAVERAGE",
            "kernel": [kernel_size, kernel_size] if isinstance(kernel_size, int) else kernel_size,
            "stride": [stride, stride] if isinstance(stride, int) else stride,
            "padding": [padding, padding] if isinstance(padding, int) else padding
        }
    })
    if DEBUG:
        print(avg_pool2d_params)
    js.data['network'].append(avg_pool2d_params)
    INLINE = False
    return x

# nn.Linear---->F.linear
def _linear(raw, input, weight, bias=None):
    x = raw(input, weight, bias)
    INLINE = True
    name = log.add_layer(name="linear_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # extract weight
    get_wgt(weight, f'{name}.weight')
    weightFile = f'{name}.weight'
    if bias is not None:
        get_wgt(bias, f'{name}.bias')
        biasFile = f'{name}.bias'
    # add json params
    linear_params = dict({
        "layerStyle": "fc",
        "layerName": name,
        "inputName": log.blobs(input),
        "weightFile": weightFile,
        "parameter":{
            "input_c": input.shape[1],
            "output_c": x.shape[1]
        }
    })
    if bias is not None:
        linear_params["biasFile"] = biasFile
    if DEBUG:
        print(linear_params)
    js.data['network'].append(linear_params)
    INLINE = False
    return x

# torch.flatten 
# one temp solution, maybe error in some scence, here just for test resnet50
# TODO: more Robust for many scence
# other way [torch.reshpe | Tensor.view <<=======>> addShuffle] also OK!
# each way can go to Rome, just follow your favority !
def _flatten(raw, input, start_dim=0, end_dim=-1):
    x = raw(input, start_dim, end_dim)
    INLINE = True
    name = log.add_layer(name="reduce(flatten)_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    if end_dim == -1:
        end_dim = 3
    #     ~Concat axis (start_dim, end_dim]
    #     ~C ~H ~W
    # 0x1  0  1  1
    # 0x2  1  0  1
    # 0x3  0  0  1
    # 0x4  1  1  0
    # 0x5  0  1  0
    # 0x6  1  0  0
    # 0x7  0  0  0
    assert start_dim >= 1
    assert end_dim <= 3
    assert end_dim > start_dim
    if(start_dim == 1 and end_dim == 3): # 1 0 0
        axes = 6
    elif(start_dim == 1 and end_dim == 2): # 1 0 1
        axes = 2
    elif(start_dim == 2 and end_dim == 3): # 1 1 0
        axes = 4
    else:
        Warning("Using torch.flatten with caution, \
            an alternative is recommended (torch.reshape | Tensor.view) <<=======>> addShuffle\n")

    # no weight extract
    # add json params
    flatten_params = dict({
        "layerStyle": "reduce",
        "layerName": name,
        "inputName": log.blobs(input),
        "type": "kAVG",
        "axes": axes,
        "keepD": True,
        })
    if DEBUG:
        print(flatten_params)
    js.data['network'].append(flatten_params)
    INLINE = False
    return x

# torch.cat
def _cat(raw, inputs, dim=0):
    x = raw(inputs, dim)
    INLINE = True
    inputName = []
    for input in inputs:
        inputName.append(log.blobs(input))
    name = log.add_layer(name="cat_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight extract
    # add json params
    cat_params = dict({
        "layerStyle": "concat",
        "layerName": name,
        "inputName": inputName,
        "axis": dim
    })
    if DEBUG:
        print(cat_params)
    js.data['network'].append(cat_params)
    INLINE = False
    return x

# nn.Softmax--->F.softmax
def _softmax(raw, input, dim=None, _stacklevel=3, dtype=None):
    x = raw(input, dim, _stacklevel, dtype)
    INLINE = True
    name = log.add_layer(name="softmax_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    #no weight to extract
    # add json params
    softmax_params = dict({"layerStyle": "softmax",
                           "layerName": name,
                           "inputName": log.blobs(input)
                           })
    if DEBUG:
        print(softmax_params)
    js.data['network'].append(softmax_params)
    INLINE = False
    return x

# ['InstanceNorm1d','InstanceNorm2d', 'InstanceNorm3d']------>F.instance_norm---->torch.instance_norm
def _instance_norm(raw, input, weight, bias, running_mean, running_var,  use_input_stats, momentum, eps, torch_backends_cudnn_enabled):
    x = raw(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, torch_backends_cudnn_enabled)
    INLINE = True
    name = log.add_layer(name="IN_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # extract weight
    get_wgt(weight, f'{name}.weight')
    get_wgt(bias, f'{name}.bias')
    # add json params
    weightFile = [f'{name}.weight', f'{name}.bias']
    instance_norm_params = dict({
        "layerStyle": "in",
        "layerName": name,
        "inputName": log.blobs(input),
        "weightFile": weightFile
    })
    if DEBUG:
        print(instance_norm_params)
    js.data['network'].append(instance_norm_params)
    INLINE = False
    return x

# ConvTranspose2d---->F.conv_transpose2d
def _conv_transpose2d(raw, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x = raw(input, weight, bias, stride, padding, output_padding, groups, dilation)
    INLINE = True
    name = log.add_layer(name="Deconv2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # extract weight
    get_wgt(weight, f'{name}.weight')
    weightFile = f'{name}.weight'
    if bias is not None:
        get_wgt(bias, f'{name}.bias')
        biasFile = f'{name}.bias'
    # add json params
    conv_transpose2d_params = dict({
        "layerStyle": "deconv",
        "layerName": name,
        "inputName": log.blobs(input),
        "weightFile": weightFile,
        "parameter":{
            "input_c": input.shape[1],
            "output_c": x.shape[1],
            "kernel": [weight.shape[2], weight.shape[3]],
            "padding": padding,
            "stride": stride
            #"output_padding": output_padding  # TRT not support, just pytorch test.
        }
    })
    if bias is not None:
        conv_transpose2d_params["biasFile"] = biasFile
    if DEBUG:
        print(conv_transpose2d_params)
    js.data['network'].append(conv_transpose2d_params)
    INLINE = False
    return x
    
# ['ConstantPad1d','ConstantPad2d', 'ConstantPad3d', 'ZeroPad2d']----->F.pad
# here for pytorch deconv outputpadding param, temp use 
# TODO: more flexible or DIY use, maybe you can use config.py? anyway it's up to you like and make sure right!
def _pad(raw, input, pad, mode="constant", value=0):
    x = raw(input, pad, mode, value)
    INLINE = True
    name = log.add_layer(name="pad_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # not weight extract
    # add json params
    pad_params = dict({
        "layerStyle": "padding",
        "layerName": name,
        "inputName": log.blobs(input),
        "parameter":{
            "input_c": input.shape[1],
            "output_c": x.shape[1],
            # 下面两个参数是为了解决反卷积outputpadding添加的，具体请联系@心满， @曹杭
            "prePadding": [0, 0], # copy from [TRT](createEng使用文档) 
            "postPadding": [1, 1] # copy from [TRT](createEng使用文档)
        }
    })
    if DEBUG:
        print(pad_params)
    js.data['network'].append(pad_params)
    INLINE = False
    return x

# torch.topk
def _topk(raw, input, k, dim=None, largest=True, sorted=True):
    x = raw(input, k, dim, largest, sorted)
    INLINE = True
    name = log.add_layer(name="topk_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weigth extract
    # add json params
    topk_params = dict({
        "layerStyle": "topk",
        "layerName": name,
        "inputName": log.blobs(input),
        "TopKOperation": "kMAX" if largest else "kMIN",
        "k": k,
        # last two params is only belong to TRT
        # TODO: more flexible, maybe use config.py to set 
        "reduceAxes": 1, # row or col
        "outputIndex": 0, # 0: values, 1: index
    })
    if DEBUG:
        print(topk_params)
    js.data['network'].append(topk_params)
    INLINE = False
    return x

# torch.argmax
def _argmax(raw, input, dim, keepdim=False):
    x = raw(input, dim, keepdim)
    INLINE = True
    name = log.add_layer(name="argmax_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight extact
    # add json params
    argmax_params = dict({
        "layerStyle": "argMax",
        "layerName": name,
        "inputName": log.blobs(input),
        # next param from [TRT](createEng 使用工具文档)， any question please contact @曹杭
        "outputName": "argMaxTestout",
        "parameter":{
            "reShape": [1, 8, 16],
            "chooseIndex": dim # from torch dim ? or just trt 2?
        }
    })
    if DEBUG:
        print(argmax_params)
    js.data['network'].append(argmax_params)
    INLINE = False
    return x

# F.interpolate ,  here we use Down/up samples the input to either the given size (resize, upsampling, downsampling)
def _interpolate(raw, input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    x = raw(input, size, scale_factor, mode, align_corners, recompute_scale_factor)
    INLINE = True
    name = log.add_layer(name="interpolate_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight extract
    # add json params
    # 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'. Default: 'nearest'
    resizeMode = {"nearest": 0, "bilinear": 1}
    interpolate_params = dict({
        "layerStyle": "resize",
        "layerName": name,
        "inputName": log.blobs(input),
        "resizeMode": resizeMode[mode],
        "alignCorners": align_corners,
        "resizeDim": size # I can't understant why it is [2, 384, 128] in [TRT](createEng使用文档)， any question please contact @曹杭
    })
    if DEBUG:
        print(interpolate_params)
    js.data['network'].append(interpolate_params)
    INLINE = False
    return x

# unaryop
def _unaryop(raw, style, input):
    x = raw(input)
    INLINE = True
    name = log.add_layer(name="unaryop_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    unaryop_params = dict({
        "layerStyle": "unary",
        "layerName": name,
        "inputName": log.blobs(input),
        "UnaryOperation": style.value
    })
    if DEBUG:
        print(unaryop_params)
    js.data['network'].append(unaryop_params)
    INLINE = False
    return x

# _add
def _add(input, *args):
    x = raw__add__(input, *args)
    name = log.add_layer(name="add_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    add_params = dict({
        "layerStyle": "eltwise",
        "layerName": name,
        "eltType": "kSUM",
        "inputName": [log.blobs(input), log.blobs(args[0])]
    })
    if DEBUG:
        print()
        print("*"*200)
        print("__add__")
        print(add_params)
    js.data['network'].append(add_params)
    return x

# _sub
def _sub(input, *args):
    x = raw__sub__(input, *args)
    name = log.add_layer(name="sub_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    sub_params = dict({
        "layerStyle": "eltwise",
        "layerName": name, 
        "eltType": "kSUB", 
        "inputName": [log.blobs(input), log.blobs(args[0])]
    })
    if DEBUG:
        print()
        print("*"*200)
        print("__sub__")
        print(sub_params)
    js.data['network'].append(sub_params)
    return x

# _expand_as
def _expand_as(input, *args):
    x = raw__expand_as__(input, *args)
    name = log.add_layer(name="expand_as_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    expand_as_params = dict({
        "layerStyle": "expand",
        "layerName": name,
        "inputName": log.blobs(input),
        "expand_as": log.blobs(args[0]) # 扩充的参照对象，目前只支持行扩厂@[TRT](createEng使用文档)
    })
    if DEBUG:
        print()
        print("*"*200)
        print("__expand_as__")
        print(expand_as_params)
    js.data['network'].append(expand_as_params)
    return x

# _permute
# TODO merge with reshape layer to shuffle layer
def _permute(input, *args):
    x = raw__permute__(input, *args)
    name = log.add_layer(name="permute_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    permute_params = dict({
        "layerStyle": "shuffle",
        "layerName": name,
        "inputName": log.blobs(input),
        "isReshape": False,
        "reshapeFirst": False,
        "reshape": None,
        "isPermute": True,
        "permute": args,
    })
    if DEBUG:
        print()
        print("*"*200)
        print("__permute__")
        print(permute_params)
    js.data['network'].append(permute_params)
    return x

# torch.div
def _div(raw, input, other):
    x = raw(input, other)
    INLINE = True
    name = log.add_layer(name="div_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    div_params = dict({
        "layerStyle": "eltwise",
        "layerName": name,
        "eltType": "kDIV",
        "inputName": [log.blobs(input), log.blobs(other)]
    })
    if DEBUG:
        print(div_params)
    js.data['network'].append(div_params)
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
    split_params = dict({
        "layerStyle": "slice",
        "layerName": layerName,
        "inputName": log.blobs(tensor),
        "axis": dim,
        "slicePoint": slicePoint[:-1]
    })
    # netscope
    if DEBUG:
        print(split_params)
    js.data['network'].append(split_params)
    INLINE = False
    return x
 
 # torch.reshape   
def _reshape(raw, input, shape):
    x = raw(input, shape)
    INLINE = True
    name = log.add_layer(name="reshape_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    reshape_params = dict({
        "layerStyle": "shuffle", 
        "layerName": name,
        "inputName":log.blobs(input),
        "isReshape": True,
        "reshapeFirst": True,
        "reshape": shape
     })
    if DEBUG:
        print(reshape_params)
    js.data['network'].append(reshape_params)
    INLINE = False
    return x
    
# torch.matmul
def _matmul(raw, input, other):
    x = raw(input, other)
    INLINE = True
    name = log.add_layer(name="matmul_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    other_matrixType = "kTRANSPOSE" if ((input.shape[1] != other.shape[0]) and (input.shape[1] == other.shape[1])) else "kNONE"
    matmul_params = dict({
        "layerStyle": "matmul",
        "layerName": name,
        "inputName": [log.blobs(input), log.blobs(other)],
        "matrixType": ["kNONE", other_matrixType]
    })
    if DEBUG:
        print(matmul_params)
    js.data['network'].append(matmul_params)
    INLINE = False
    return x



# nn.AdaptiveAvgPool2d----->F.adaptive_avg_pool2d
# tensorrt not support , just pytorch test
def _adaptive_avg_pool2d(raw, input, output_size):
    x = raw(input, output_size)
    INLINE = True
    name = log.add_layer(name="adaptive_avg_pool2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight extract
    # add json params
    adaptive_avg_pool2d_params = dict({
        "layerStyle": "pool",
        "layerName": name,
        "inputName": log.blobs(input),
        "parameter":{
            "poolType": "kADAPTIVEAVGPOOL2D",
            "output_size": [output_size, output_size] if isinstance(output_size, int) else output_size
        }
    })
    if DEBUG:
        print(adaptive_avg_pool2d_params)
    js.data['network'].append(adaptive_avg_pool2d_params)
    INLINE = False
    return x

# a[11 dim weights] * b[tensor] for scale layer, just test
# TODO: You can add it manually after generating JSON. and wgt. This is just a format note
# or you can complete it by yourself way , maybe config.py can be useful?
def _mul(input, *args):
    x = raw__mul__(input, *args)
    name = log.add_layer(name="mul_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    mul_params = dict({
        "layerStyle": "scale",
        "scaleMode": "kELEMENTWISE", # "kUNIFORM", "kCHANNEL", "kELEMENTWISE"
        "layerName": name,
        "inputName": log.blobs(input),
        "weightFile": ["shift_weights", "scale_weights", "power_weights"],
        "outputName": "att_score"
    })
    if DEBUG:
        print()
        print("*"*200)
        print("__mul__")
        print(mul_params)
    js.data['network'].append(mul_params)
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
            if DEBUG:
                INREG = False
                print()
                print("*"*200)
                for stack in traceback.walk_stack(None):
                    if 'self' in stack[0].f_locals:
                        layer = stack[0].f_locals['self']
                        if type(layer).__name__ in REGISTERED_LIST:
                            INREG = True
                            print(layer)
                            break
                if not INREG:
                    INREG = False
                    print(self.raw.__name__)
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
            if DEBUG:
                INREG = False
                print()
                print("*"*200)
                for stack in traceback.walk_stack(None):
                    if 'self' in stack[0].f_locals:
                        layer = stack[0].f_locals['self']
                        if type(layer).__name__ in REGISTERED_LIST:
                            INREG = True
                            print(layer)
                            break
                if not INREG:
                    INREG = False
                    print(self.raw.__name__)
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
# torch op
torch.batch_norm = Rp(torch.batch_norm, _batch_norm)
torch.sigmoid = Rp(torch.sigmoid, _sigmoid)
torch.flatten = Rp(torch.flatten, _flatten)
torch.cat = Rp(torch.cat, _cat)
torch.instance_norm = Rp(torch.instance_norm, _instance_norm)
torch.topk = Rp(torch.topk, _topk)
torch.argmax = Rp(torch.argmax, _argmax)
torch.matmul = Rp(torch.matmul, _matmul)
torch.div = Rp(torch.div, _div) # for [TRT] elt layer's kDIV op
torch.split = Rp(torch.split, _split) # for [TRT] slice layer
torch.reshape = Rp(torch.reshape, _reshape) # instead view for [TRT] shuffle layer

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
    # c = a * b # for [TRT] scale layer
    raw__mul__ = t.__mul__
    t.__mul__ = _mul

    # view(instead by torch.reshape), permute for [TRT] shuffle layer
    raw__permute__ = t.permute
    t.permute = _permute
    # expand_as for [TRT] expand layer
    raw__expand_as__ = t.expand_as
    t.expand_as = _expand_as

    


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
    if len(outs) >=2:
        for i, out in enumerate(outs):
            for j, layer_param in enumerate(js.data['network']):
                if layer_param['layerName'] == LayerOut_id[int(id(out))]:
                    js.data['network'][j]["outputName"] = f"{layer_param['layerName']}_{i+1}"
    elif len(outs)==1:
        for j, layer_param in enumerate(js.data['network']):
            if layer_param['layerName'] == LayerOut_id[int(id(outs))]:
                js.data['network'][j]["outputName"] = f"{layer_param['layerName']}_1"
                break

    js.dump()
    print()
    print("Successed !")
    return