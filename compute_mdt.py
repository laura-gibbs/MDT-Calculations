import math
import numpy as np
import os
import array
import matplotlib.pyplot as plt
from utils import define_dims, create_coords, bound_arr
from read_data import read_surface, read_surfaces, write_surface, apply_mask


def calculate_mdt(mss, geoid, mask=True):
    mdt = mss - geoid
    if mask:
        return apply_mask(0.25, mdt)
    else:
        return mdt


def main():
    # path1 = './data/src/'
    path2 = './data/res/'
    cmippath = './cmip5/rcp60/'

    rcp60_mdts = read_surfaces('cmip5_rcp60_mdts_yr5.dat', cmippath, number=3,
                               start=100)
    example_mdt = bound_arr(rcp60_mdts[0].T, -1.5, 1.5)
    # plt.imshow(example_mdt)
    # plt.show()

    mdt = read_surface('dip_1000_dtu15gtim5do0280_rr0004.dat', path2, transpose=True)
    plt.imshow(mdt)
    plt.show()



    # geoid = read_surface('gtim5.dat', res, path1, nans=False)
    # mdt = calculate_mdt(mss, geoid)
    # calculate_mdt(apply_mask(res, mss, geoid))


if __name__ == '__main__':
    main()
