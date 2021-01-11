# copy from https://github.com/lukemelas/EfficientNet-PyTorch
__version__ = "0.7.0"
from .model import EfficientNet, VALID_MODELS
from .model import efficientnet_b0, efficientnet_b0_advprop, \
                                               efficientnet_b1, efficientnet_b1_advprop, \
                                               efficientnet_b2, efficientnet_b2_advprop, \
                                               efficientnet_b3, efficientnet_b3_advprop, \
                                               efficientnet_b4, efficientnet_b4_advprop, \
                                               efficientnet_b5, efficientnet_b5_advprop, \
                                               efficientnet_b6, efficientnet_b6_advprop, \
                                               efficientnet_b7, efficientnet_b7_advprop, \
                                               efficientnet_b8, efficientnet_b8_advprop, \
                                               efficientnet_l2, efficientnet_l2_advprop
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)
