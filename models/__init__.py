from .alexnet import alexnet
from .vggnet import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from .googlenet import googlenet
from .inception import inception_v3
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from .densenet import densenet121, densenet161, densenet169, densenet201
from .squeezenet import squeezenet1_0, squeezenet1_1
from .shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from .mobilenetv2 import mobilenet_v2
from .mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from .mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from .efficientnet import efficientnet_b0, efficientnet_b0_advprop, efficientnet_b1, efficientnet_b1_advprop, efficientnet_b2, efficientnet_b2_advprop, efficientnet_b3, efficientnet_b3_advprop, efficientnet_b4, efficientnet_b4_advprop, \
                          efficientnet_b5, efficientnet_b5_advprop, efficientnet_b6, efficientnet_b6_advprop, efficientnet_b7, efficientnet_b7_advprop, efficientnet_b8, efficientnet_b8_advprop, efficientnet_l2, efficientnet_l2_advprop                                                          
from .hrnet import hrnet_w18, hrnet_w18_small_v1, hrnet_w18_small_v2, \
                    hrnet_w30, hrnet_w32, hrnet_w40, hrnet_w44, hrnet_w48, hrnet_w64
from .ghostnet import ghostnet_1x
from .res2net import res2net_dla60, res2next_dla60, \
                    res2net50_v1b, res2net101_v1b, res2net50_v1b_26w_4s, res2net101_v1b_26w_4s, res2net152_v1b_26w_4s, \
                    res2net50, res2net50_26w_4s, res2net101_26w_4s, res2net50_26w_6s, res2net50_26w_8s, res2net50_48w_2s, res2net50_14w_8s, \
                    res2next50
from .regnet import regnet_200M, regnet_400M, regnet_600M, regnet_800M, regnet_1600M, regnet_3200M, regnet_4000M, regnet_6400M
from .resnest import resnest50, resnest101, resnest200, resnest269, \
                     resnest50_fast_1s1x64d, resnest50_fast_2s1x64d, resnest50_fast_4s1x64d, \
                     resnest50_fast_1s2x40d, resnest50_fast_2s2x40d, resnest50_fast_4s2x40d, \
                     resnest50_fast_1s4x24d

__all__ = dict({"alexnet": alexnet,
                "vgg11": vgg11, "vgg11_bn": vgg11_bn, 
                "vgg13": vgg13, "vgg13_bn": vgg13_bn, 
                "vgg16": vgg16, "vgg16_bn": vgg16_bn,
                "vgg19": vgg19, "vgg19_bn": vgg19_bn,
                "googlenet": googlenet,
                "inception_v3": inception_v3,
                "resnet18": resnet18,
                "resnet34": resnet34,
                "resnet50": resnet50,
                "resnet101": resnet101,
                "resnet152": resnet152,
                "resnext50_32x4d": resnext50_32x4d,
                "resnext101_32x8d": resnext101_32x8d,
                "wide_resnet50_2": wide_resnet50_2,
                "wide_resnet101_2": wide_resnet101_2,
                "densenet121": densenet121,
                "densenet169": densenet169,
                "densenet161": densenet161,
                "densenet201": densenet201,
                "squeezenet1_0": squeezenet1_0,
                "squeezenet1_1": squeezenet1_1,
                "shufflenet_v2_x0_5": shufflenet_v2_x0_5,
                "shufflenet_v2_x1_0": shufflenet_v2_x1_0,
                "shufflenet_v2_x1_5": shufflenet_v2_x1_5,
                "shufflenet_v2_x2_0": shufflenet_v2_x2_0,
                "mobilenet_v2": mobilenet_v2,
                "mnasnet0_5": mnasnet0_5,
                "mnasnet0_75": mnasnet0_75,
                "mnasnet1_0": mnasnet1_0,
                "mnasnet1_3": mnasnet1_3,
                "efficientnet_b0": efficientnet_b0,
                "efficientnet_b0_advprop": efficientnet_b0_advprop,
                "efficientnet_b1": efficientnet_b1,
                "efficientnet_b2": efficientnet_b2,
                "efficientnet_b3": efficientnet_b3,
                "efficientnet_b4": efficientnet_b4,
                "efficientnet_b5": efficientnet_b5,
                "efficientnet_b6": efficientnet_b6,
                "efficientnet_b7": efficientnet_b7,
                "efficientnet_b8": efficientnet_b8,
                "efficientnet_b1_advprop": efficientnet_b1_advprop,
                "efficientnet_b2_advprop": efficientnet_b2_advprop,
                "efficientnet_b3_advprop": efficientnet_b3_advprop,
                "efficientnet_b4_advprop": efficientnet_b4_advprop,
                "efficientnet_b5_advprop": efficientnet_b5_advprop,
                "efficientnet_b6_advprop": efficientnet_b6_advprop,
                "efficientnet_b7_advprop": efficientnet_b7_advprop,
                "efficientnet_b8_advprop": efficientnet_b8_advprop,
                "efficientnet_l2": efficientnet_l2, 
                "efficientnet_l2_advprop": efficientnet_l2_advprop,
                "hrnet_w18": hrnet_w18,
                "hrnet_w18_small_v1": hrnet_w18_small_v1,
                "hrnet_w18_small_v2": hrnet_w18_small_v2,
                "hrnet_w30": hrnet_w30,
                "hrnet_w32": hrnet_w32,
                "hrnet_w40": hrnet_w40,
                "hrnet_w44": hrnet_w44,
                "hrnet_w48": hrnet_w48,
                "hrnet_w64": hrnet_w64,
                "ghostnet_1x": ghostnet_1x,
                "mobilenetv3_large": mobilenetv3_large,
                "mobilenetv3_small": mobilenetv3_small,
                "res2net_dla60": res2net_dla60, 
                "res2next_dla60": res2next_dla60,
                "res2net50_v1b":res2net50_v1b, 
                "res2net101_v1b": res2net101_v1b, 
                "res2net50_v1b_26w_4s": res2net50_v1b_26w_4s, 
                "res2net101_v1b_26w_4s": res2net101_v1b_26w_4s, 
                "res2net152_v1b_26w_4s": res2net152_v1b_26w_4s,
                "res2net50": res2net50, 
                "res2net50_26w_4s": res2net50_26w_4s, 
                "res2net101_26w_4s": res2net101_26w_4s, 
                "res2net50_26w_6s": res2net50_26w_6s, 
                "res2net50_26w_8s": res2net50_26w_8s, 
                "res2net50_48w_2s": res2net50_48w_2s, 
                "res2net50_14w_8s": res2net50_14w_8s,
                "res2next50": res2next50,
                "regnet_200M": regnet_200M, 
                "regnet_400M": regnet_400M, 
                "regnet_600M": regnet_600M, 
                "regnet_800M": regnet_800M, 
                "regnet_1600M": regnet_1600M, 
                "regnet_3200M": regnet_3200M, 
                "regnet_4000M": regnet_4000M, 
                "regnet_6400M": regnet_6400M,
                "resnest50": resnest50, 
                "resnest101": resnest101, 
                "resnest200": resnest200, 
                "resnest269": resnest269, 
                "resnest50_fast_1s1x64d": resnest50_fast_1s1x64d, 
                "resnest50_fast_2s1x64d": resnest50_fast_2s1x64d, 
                "resnest50_fast_4s1x64d": resnest50_fast_4s1x64d,
                "resnest50_fast_1s2x40d": resnest50_fast_1s2x40d, 
                "resnest50_fast_2s2x40d": resnest50_fast_2s2x40d, 
                "resnest50_fast_4s2x40d": resnest50_fast_4s2x40d, 
                "resnest50_fast_1s4x24d": resnest50_fast_1s4x24d
                })