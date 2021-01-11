# copy from https://github.com/HRNet/HRNet-Image-Classification
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
from .model import hrnet_w18, hrnet_w18_small_v1, hrnet_w18_small_v2, \
                    hrnet_w30, hrnet_w32, hrnet_w40, hrnet_w44, hrnet_w48, hrnet_w64