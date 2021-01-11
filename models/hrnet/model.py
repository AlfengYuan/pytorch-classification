"""
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
`https://arxiv.org/pdf/1908.07919.pdf`
"""
import torch
import torch.nn as nn
from .default import _C as config
from .default import update_config
from .cls_hrnet import get_cls_net
from torch.hub import load_state_dict_from_url
model_cfg = {
    "hrnet_w18": "models/hrnet/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
    "hrnet_w18_url": "https://www.alfeng.icu/download/hrnetv2_w18_imagenet_pretrained.pth",
    "hrnet_w18_small_v1": "models/hrnet/experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
    "hrnet_w18_small_v1_url": "https://www.alfeng.icu/download/hrnet_w18_small_model_v1.pth",
    "hrnet_w18_small_v2": "models/hrnet/experiments/cls_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
    "hrnet_w18_small_v2_url": "https://www.alfeng.icu/download/hrnet_w18_small_model_v2.pth",
    "hrnet_w30":"models/hrnet/experiments/cls_hrnet_w30_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
    "hrnet_w30_url": "https://www.alfeng.icu/download/hrnetv2_w30_imagenet_pretrained.pth",
    "hrnet_w32": "models/hrnet/experiments/cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
    "hrnet_w32_url": "https://www.alfeng.icu/download/hrnetv2_w32_imagenet_pretrained.pth",
    "hrnet_w40": "models/hrnet/experiments/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
    "hrnet_w40_url": "https://www.alfeng.icu/download/hrnetv2_w40_imagenet_pretrained.pth",
    "hrnet_w44": "models/hrnet/experiments/cls_hrnet_w44_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
    "hrnet_w44_url": "https://www.alfeng.icu/download/hrnetv2_w44_imagenet_pretrained.pth",
    "hrnet_w48": "models/hrnet/experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
    "hrnet_w48_url": "https://www.alfeng.icu/download/hrnetv2_w48_imagenet_pretrained.pth",
    "hrnet_w64": "models/hrnet/experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
    "hrnet_w64_url": "https://www.alfeng.icu/download/hrnetv2_w64_imagenet_pretrained.pth"
} 

def _hrnet(pretrained, checkpoints, progress, **kwargs):
    update_config(config, model_cfg[kwargs['model_name']])
    model = get_cls_net(config)
    if pretrained:
        if checkpoints is not None:
            model.load_state_dict(torch.load(checkpoints), True)
            return model
        state_dict = load_state_dict_from_url(model_cfg[kwargs['model_name'] + '_url'])
        model.load_state_dict(state_dict)
        return model
    for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    return model

def hrnet_w18(pretrained=False, checkpoints=None, progress=True, **kwargs):
    kwargs['model_name'] = "hrnet_w18"
    return _hrnet(pretrained, checkpoints, progress, **kwargs)

def hrnet_w18_small_v1(pretrained=False, checkpoints=None, progress=True, **kwargs):
    kwargs['model_name'] = "hrnet_w18_small_v1"
    return _hrnet(pretrained, checkpoints, progress, **kwargs)

def hrnet_w18_small_v2(pretrained=False, checkpoints=None, progress=True, **kwargs):
    kwargs['model_name'] = "hrnet_w18_small_v2"
    return _hrnet(pretrained, checkpoints, progress, **kwargs)

def hrnet_w30(pretrained=False, checkpoints=None, progress=True, **kwargs):
    kwargs['model_name'] = "hrnet_w30"
    return _hrnet(pretrained, checkpoints, progress, **kwargs)

def hrnet_w32(pretrained=False, checkpoints=None, progress=True, **kwargs):
    kwargs['model_name'] = "hrnet_w32"
    return _hrnet(pretrained, checkpoints, progress, **kwargs)

def hrnet_w40(pretrained=False, checkpoints=None, progress=True, **kwargs):
    kwargs['model_name'] = "hrnet_w40"
    return _hrnet(pretrained, checkpoints, progress, **kwargs)

def hrnet_w44(pretrained=False, checkpoints=None, progress=True, **kwargs):
    kwargs['model_name'] = "hrnet_w44"
    return _hrnet(pretrained, checkpoints, progress, **kwargs)

def hrnet_w48(pretrained=False, checkpoints=None, progress=True, **kwargs):
    kwargs['model_name'] = "hrnet_w48"
    return _hrnet(pretrained, checkpoints, progress, **kwargs)

def hrnet_w64(pretrained=False, checkpoints=None, progress=True, **kwargs):
    kwargs['model_name'] = "hrnet_w64"
    return _hrnet(pretrained, checkpoints, progress, **kwargs)