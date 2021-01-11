# copy from https://github.com/zhanghang1989/ResNeSt
from .resnest import resnest50, resnest101, resnest200, resnest269
from .ablation import resnest50_fast_1s1x64d, resnest50_fast_2s1x64d, resnest50_fast_4s1x64d, \
                        resnest50_fast_1s2x40d, resnest50_fast_2s2x40d, resnest50_fast_4s2x40d, \
                        resnest50_fast_1s4x24d
