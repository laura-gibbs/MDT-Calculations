import math
import numpy as np
import os
import array
import matplotlib.pyplot as plt
from utils import define_dims, create_coords, bound_arr
# from read_data import read_surface, write_surface, apply_mask


def calculate_mdt(mss, geoid, mask=True):
    mdt = mss - geoid
    if mask:
        return apply_mask(0.25, mdt)
    else:
        return mdt


def read_surfaces(filename, resolution=None, path=None, fortran=True,
                 nans=True, transpose=False):
    r"""
    """
    if resolution is None:
        resolution = parse_res(filename)

    II, JJ = define_dims(resolution)
    print(II, JJ, resolution)

    if path is None:
        path = ""

    filepath = os.path.join(os.path.normpath(path), filename)
    fid = open(filepath, mode='rb')
    mdts = []
    # return

    # Loads Fortran array (CxR) or Python array (RxC)
    if fortran:
        buffer = fid.read(4)
        while buffer != b'':
            size = np.frombuffer(buffer, dtype=np.int32)[0]
            floats = np.asfortranarray(np.frombuffer(fid.read(size), dtype=np.float32))
            floats = np.array(floats)
            floats = np.reshape(floats, (II, JJ), order='F')
            mdts.append(floats)
            print(f'Loaded MDT #{len(mdts)}')
            footer_value = np.frombuffer(fid.read(4), dtype=np.int32)[0]
            buffer = fid.read(4)
    else:
        floats = np.frombuffer(fid.read(), dtype=np.float32)
        floats = floats[1:len(floats)-1]
        floats = np.asarray(floats)
        floats = np.reshape(floats, (II, JJ))

    mdts = np.array(mdts)
    if nans:
        mdts[mdts <= -1.7e7] = np.nan
    if transpose:
        return np.transpose(mdts, (0, 2, 1))

    return mdts


def main():
    res = 0.25
    II, JJ = define_dims(res)
    
    path1 = './data/src/'
    path2 = './data/res/'
    cmippath = './cmip5/rcp60/'
    pathout = './data/res/'

    rcp60_mdts = read_surfaces('cmip5_rcp60_mdts_yr5.dat', res, cmippath)
    print(rcp60_mdts.shape)
    # geoid = read_surface('gtim5.dat', res, path1, nans=False)
    # mdt = calculate_mdt(mss, geoid)
    # calculate_mdt(apply_mask(res, mss, geoid))
    example_mdt = bound_arr(rcp60_mdts[664].T, -1.5, 1.5)
    print(example_mdt[0, 0])
    plt.imshow(example_mdt)
    plt.show()


if __name__ == '__main__':
    main()
