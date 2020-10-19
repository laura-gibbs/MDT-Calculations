import math
import numpy as np
import os
import array
import matplotlib.pyplot as plt
from utils import define_dims, create_coords, bound_arr
from read_data import read_surface, write_surface, apply_mask


def calculate_mdt(mss, geoid, mask=True):
    mdt = mss - geoid
    if mask:
        return apply_mask(0.25, mdt)
    else:
        return mdt


def main():
    res = 0.25
    II, JJ = define_dims(res)
    
    path1 = './data/src/'
    path2 = './data/res/'
    cmippath = './cmip5/rcp45/'
    pathout = './data/res/'

    rcp45_mdt = read_surface('cmip5_rcp45_mdts_yr5.dat', res, cmippath)
    # geoid = read_surface('gtim5.dat', res, path1, nans=False)
    # mdt = calculate_mdt(mss, geoid)
    # calculate_mdt(apply_mask(res, mss, geoid))
    rcp45_mdt = bound_arr(rcp45_mdt.T, -1.5, 1.5)
    print(rcp45_mdt)
    plt.plot(rcp45_mdt)
    plt.show()


if __name__ == '__main__':
    main()
