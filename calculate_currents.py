import math
import numpy as np
import os
import array
import matplotlib.pyplot as plt
from sys import platform
import read_data as rd
from utils import define_dims, create_coords

r = 6371229.0
omega = 7.29e-5
g = 9.80665


def mdt_cs(II, JJ, lat, mdt, cs):
    r"""
    """
    # Define parameters
    lats_r = np.deg2rad(lat[1] - lat[0])
    lons_r = lats_r / 2

    # Calculate zonal width of a grid cell (m) (depends on Latitude)
    dx = np.array([2 * r * lons_r * math.cos(np.deg2rad(lat[j])) for j in range(JJ)])

    # Calculate meridional width of grid cell (m) (does not depend on lat)
    dy = r * lats_r

    # Calculate coriolis parameter f
    f0 = np.array([2.0 * omega * math.sin(np.deg2rad(lat[j])) for j in range(JJ)])

    u = np.zeros((II, JJ))
    v = np.zeros((II, JJ))

    # Compute currents
    print(f'mdt.shape={mdt.shape}')
    for j in range(1, JJ-1):
        for i in range(II):
            if not np.isnan(mdt[i, j]) and not np.isnan(mdt[i, j-1]):
                u[i, j] = -(g / f0[j]) * (mdt[i, j] - mdt[i, j-1]) / (dy)

            if not np.isnan(mdt[i, j]) and not np.isnan(mdt[i-1, j]):
                v[i, j] = (g / f0[j]) * (mdt[i, j] - mdt[i-1, j]) / (dx[j])

            cs[i, j] = math.sqrt(u[i, j] ** 2 + v[i, j] ** 2)

    return cs, u, v


def main():
    i1 = 0
    i2 = 1440
    j1 = 0
    j2 = 720
    II = i2 - i1
    JJ = j2 - j1
    IIin = 1440
    JJin = 720
    upper_bd = 1.5
    lower_bd = -1.5

    if platform == "darwin":
        path0 = './fortran/data/src'
        path1 = './fortran/data/res'
        path2 = './fortran/data/test'
    elif platform == "win32":
        path0 = '.\\fortran\\data\\src\\'
        path1 = '.\\fortran\\data\\res\\'
        path2 = '.\\fortran\\data\\test\\'
    
    # Define global lon and lat (lat shifted to midpts set between -90 and 90)
    glon = np.array([0.25 * (i - 0.5) for i in range(IIin)])
    glat = np.array([0.25 * (j - 0.5) - 90.0 for j in range(JJin)])
    lats = np.deg2rad(0.25)
    lat0 = np.deg2rad(-89.875)
    
    gcs = np.zeros((IIin, JJin))


    gmdt = rd.read_dat('shmdtfile.dat', path=path1, shape=(IIin, JJin), nans=True, transpose=False)
    mask = rd.read_dat('mask_glbl_qrtd.dat', path=path0, shape=(IIin, JJin), nans=True, transpose=False)
    gcs, u, v = mdt_cs(IIin, JJin, glat, gmdt, gcs)

    gmdt = gmdt + mask
    gcs = gcs + mask

    mdt = gmdt[i1:i2, j1:j2]
    cs = gcs[i1:i2, j1:j2]

    glat_rad = np.deg2rad(glat)

    ds = np.array([(r ** 2 * lats) * (math.sin(glat_rad[j]) -
                  math.sin(glat_rad[j-1])) for j in range(JJ)])

    # ds = np.array([0.5 * (r * lats) ** 2 * (math.cos(glat_rad[j]) +
    #              math.cos(glat_rad[j-1])) for j in range(JJ)])

    sum_mdt_ds = 0.
    ocean_area = 0.
    for i in range(II):
        for j in range(JJ):
            if not np.isnan(mdt[i, j]):
                sum_mdt_ds += mdt[i, j] * ds[j]
                ocean_area += ds[j]

    mn = sum_mdt_ds / ocean_area
    mdt = mdt - mn
    print(ocean_area, sum_mdt_ds, mn)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    mdt = mdt.T
    mdt[np.isnan(mdt)] = lower_bd
    mdt[mdt < lower_bd] = lower_bd
    mdt[mdt > upper_bd] = upper_bd
    
    cs = cs.T
    cs[np.isnan(cs)] = lower_bd
    cs[cs < lower_bd] = lower_bd
    cs[cs > upper_bd] = upper_bd
    
    ax1.imshow(mdt)
    ax2.imshow(cs)
    plt.show() 

    # write_dat(path2, 'tmp.dat', gmdt)
    # write_dat(path2, 'tmp2.dat', mdt)
    # write_dat(path2, 'shmdtout.dat', cs)


if __name__ == '__main__':
    main()