from Auto_Json import config as cfg
import struct
import numpy as np
import os

def get_wgt(weight, name):
    # this function for generating .wgt weights for each layer.
    ts = weight.cpu().detach().numpy().copy()
    shape = ts.shape
    size = shape
    allsize = 1
    for idx in range(len(size)):
        allsize *= size[idx]
    ts = ts.reshape(allsize)
    with open(os.path.join(cfg.WEIGHTS_DIR, "{}.wgt".format(name)), "wb") as f:
        a = struct.pack('i', allsize)
        f.write(a)
        for i in range(allsize):
            a = struct.pack('f', ts[i])
            f.write(a)

