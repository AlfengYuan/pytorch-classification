# copy from https://github.com/yhhhli/RegNet-Pytorch
# paper: `https://arxiv.org/pdf/2003.13678.pdf`
"""
@InProceedings{Radosavovic_2020_CVPR,
author = {Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
title = {Designing Network Design Spaces},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
"""
from .regnet import regnet_200M, regnet_400M, regnet_600M, regnet_800M, regnet_1600M, regnet_3200M, regnet_4000M, regnet_6400M