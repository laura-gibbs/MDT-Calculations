import math
import numpy as np
import matplotlib.pyplot as plt
from read_data import read_surface, write_surface
from utils import define_dims, create_coords, bound_arr

r = 6371229.0
omega = 7.29e-5
g = 9.80665


def calc_currents(resolution, mdt):
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


def centralise_mdt(resolution, mdt):
    r"""
    """
    longitude, latitude = create_coords(resolution)
    ds = grid_square_area(resolution, latitude)
    mn = calc_mean(mdt, ds)
    mdt = centralise_data(mdt, mn)
    return mdt


def apply_mask(resolution, surface, mask_filename=None, path=None):
    if mask_filename is None and path is None:
        mask = read_surface('mask_glbl_qrtd.dat', resolution,
                            './fortran/data/src', True,
                            False)
    else:
        mask = read_surface(mask_filename, resolution, path, True,
                            False)
    surface = surface + mask
    return surface


def fn_name(resolution, surface_path, surface_filename):
    gmdt = apply_mask(
            resolution,
            read_surface(
                surface_filename,
                resolution,
                surface_path,
                True,
                False
            ))

    currents, u, v = calc_currents(resolution, gmdt)
    currents = apply_mask(resolution, currents)
    mdt = centralise_mdt(resolution, gmdt)
    return mdt, currents


def main():
    res = 0.25
    # II, JJ = define_dims(res)

    # path0 = './fortran/data/src'
    path1 = './fortran/data/res'
    # path2 = './fortran/data/test'
    mdt_filename = 'shmdtfile.dat'

    mdt, cs = fn_name(res, path1, mdt_filename)

    # gmdt = apply_mask(res, read_surface(mdt_filename, resolution=res,
    #                   path=path1, nans=True, transpose=False))

    # gcs, u, v = calc_currents(res, gmdt)
    # gcs = apply_mask(res, gcs)

    # mdt = centralise_mdt(res, gmdt)
    # mdt = bound_arr(mdt.T, -1.5, 1.5)
    # cs = bound_arr(gcs.T, -1.5, 1.5)

    # ds = grid_square_area(res, glat)
    # mn = calc_mean(mdt, ds)
    # # print(calc_ocean_area(mdt, ds), sum_mdt_ds(mdt, ds), mn)
    # mdt = centralise_data(mdt, mn)
    
    mdt = bound_arr(mdt.T, -1.5, 1.5)
    cs = bound_arr(cs.T, -1.5, 1.5)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(mdt)
    ax2.imshow(cs)
    plt.show()

    # write_surface(path2, 'tmp.dat', gmdt)
    # write_surface(path2, 'tmp2.dat', mdt)
    # write_surface(path2, 'shmdtout.dat', cs)


if __name__ == '__main__':
    main()
