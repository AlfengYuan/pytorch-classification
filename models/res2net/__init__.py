# copy from https://github.com/Res2Net/Res2Net-PretrainedModels
"""
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2020},
  doi={10.1109/TPAMI.2019.2938758}, 
}
"""
from .dla import res2net_dla60, res2next_dla60
from .res2net_v1b import res2net50_v1b, res2net101_v1b, res2net50_v1b_26w_4s, res2net101_v1b_26w_4s, res2net152_v1b_26w_4s
from .res2net import res2net50, res2net50_26w_4s, res2net101_26w_4s, res2net50_26w_6s, res2net50_26w_8s, res2net50_48w_2s, res2net50_14w_8s
from .res2next import res2next50