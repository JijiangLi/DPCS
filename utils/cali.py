import torch
import numpy as np
import yaml
from utils.fov import *
"""
This file was clipped from DeProCams https://github.com/BingyaoHuang/DeProCams/blob/main/src/python/utils.py
aims to load calibration data from params.yaml into the geometry of the pro-cam like extrinsics and intrinsics 
into mistuba3 rendering configuration.

"""
# convert yaml file string to np array
def stringToMat(m):
    n_rows = len(m)
    n_cols = len(m[0].split(','))

    mat = np.zeros((n_rows, n_cols))

    for r in range(n_rows):
        cur_row = m[r].split(',')
        for c in range(n_cols):
            mat[r][c] = float(cur_row[c])

    return mat

# load yaml to tensor

def loadCalib(file_name):
    with open(file_name) as f:
        raw_data = yaml.load(f, yaml.Loader)

    calib_data = {}
    for m in raw_data:
        calib_data[m] = torch.Tensor(stringToMat(raw_data[m]))
    # calib_data['camRT'][0:3,0:3] = torch.eye(3,3)

    # convert to Kornia Bx4x4
    tensor_4x4 = torch.eye(4, 4)
    tensor_4x4[0:3, 0:3] = calib_data['camK']
    calib_data['camK'] = tensor_4x4.unsqueeze(0).clone()
    tensor_4x4[0:3, 0:3] = calib_data['prjK']
    calib_data['prjK'] = tensor_4x4.unsqueeze(0).clone()

    # extrinsics 3x4 ->1x4x4
    tensor_4x4[0:3, ...] = calib_data['camRT']
    calib_data['camRT'] = tensor_4x4.unsqueeze(0).clone()
    tensor_4x4[0:3, ...] = calib_data['prjRT']
    calib_data['prjRT'] = tensor_4x4.unsqueeze(0).clone()
    calib_data["max_depth"] = 3

    return calib_data

