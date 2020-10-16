import math
import numpy as np
import matplotlib.pyplot as plt
import read_data as rd
from utils import define_dims, create_coords, bound_arr

r = 6371229.0
omega = 7.29e-5
g = 9.80665


def mdt_cs(resolution, mdt):
    r"""
    """
    # Define parameters
    II, JJ = define_dims(resolution)
    longitude, latitude = create_coords(resolution)
    lats_r = np.deg2rad(latitude[1] - latitude[0])
    lons_r = lats_r / 2
    cs = np.zeros((II, JJ))

    # Calculate zonal width of a grid cell (m) (depends on Latitude)
    dx = np.array([2 * r * lons_r * math.cos(np.deg2rad(latitude[j]))
                  for j in range(JJ)])

    # Calculate meridional width of grid cell (m) (does not depend on lat)
    dy = r * lats_r

    # Calculate coriolis parameter f
    f0 = np.array([2.0 * omega * math.sin(np.deg2rad(latitude[j]))
                  for j in range(JJ)])

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


def grid_square_area(resolution, latitude):
    II, JJ = define_dims(resolution)
    lats = np.deg2rad(resolution)
    longitude, latitude = create_coords(resolution)
    glat_rad = np.deg2rad(latitude)
    ds = np.array([(r ** 2 * lats) * (math.sin(glat_rad[j]) -
                  math.sin(glat_rad[j-1])) for j in range(JJ)])

    # ds = np.array([0.5 * (r * lats) ** 2 * (math.cos(glat_rad[j]) +
    #              math.cos(glat_rad[j-1])) for j in range(JJ)])
    return ds


def calc_ocean_area(mdt, ds):
    r"""
    Calculates ocean area from MDT and grid square area
    """
    area = 0.
    II, JJ = mdt.shape[0], mdt.shape[1]
    for i in range(II):
        for j in range(JJ):
            if not np.isnan(mdt[i, j]):
                area += ds[j]

    return area


def sum_mdt_ds(mdt, ds):
    r"""
    Approximates ocean volume from MDT and grid square area?
    """
    sm = 0.
    II, JJ = mdt.shape[0], mdt.shape[1]
    for i in range(II):
        for j in range(JJ):
            if not np.isnan(mdt[i, j]):
                sm += mdt[i, j] * ds[j]

    return sm

    # calc_ocean_area = np.sum(ds * (1 - mask))


def calc_mean(mdt, ds):
    return sum_mdt_ds(mdt, ds) / calc_ocean_area(mdt, ds)


def centralise_data(arr, mn):
    return arr - mn

# def centre_mdt
# calls helper fn to calc ocean area and volume
# also uses centralise_data fn


# def get_mask(mask_filename):

def main():
    res = 0.25
    II, JJ = define_dims(res)

    path0 = './fortran/data/src'
    path1 = './fortran/data/res'
    # path2 = './fortran/data/test'

    glon, glat = create_coords(res)
    # lats = np.deg2rad(res)
    # lat0 = np.deg2rad(-89.875)

    gmdt = rd.read_dat('shmdtfile.dat', resolution=res, path=path1, nans=True,
                       transpose=False)
    mask = rd.read_dat('mask_glbl_qrtd.dat', resolution=res, path=path0,
                       nans=True, transpose=False)
    gcs, u, v = mdt_cs(res, gmdt)

    gmdt = gmdt + mask
    gcs = gcs + mask

    mdt = gmdt
    cs = gcs

    # glat_rad = np.deg2rad(glat)

    ds = grid_square_area(res, glat)

    mn = calc_mean(mdt, ds)
    # print(calc_ocean_area(mdt, ds), sum_mdt_ds(mdt, ds), mn)
    mdt = centralise_data(mdt, mn)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    mdt = bound_arr(mdt.T, -1.5, 1.5)
    cs = bound_arr(cs.T, -1.5, 1.5)

    ax1.imshow(mdt)
    ax2.imshow(cs)
    plt.show()

    # write_dat(path2, 'tmp.dat', gmdt)
    # write_dat(path2, 'tmp2.dat', mdt)
    # write_dat(path2, 'shmdtout.dat', cs)


if __name__ == '__main__':
    main()
