import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from netCDF4 import Dataset as netcdf_dataset
from cartopy import config
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import turbo_colormap_mpl
from utils import bound_arr
import cartopy.feature as cfeature
from read_data import read_surface, read_surfaces, write_surface, apply_mask

def cbar_fmt(x, pos):
    r"""Colorbar function, formats colorbar labels to include units 'm'.

    Args:
        x (float): colorbar label value
        pos (int): position of colorbar label

    Returns:
        str(x) + 'm' (String)

    """
    return str(x) + 'm'


def main():
    crs = ccrs.PlateCarree(central_longitude=0)
    x
    fname = ('../DTU/DTU10MDT_2min.nc')
    dataset = netcdf_dataset(fname)

    # print(dataset.variables)
    mdt = dataset.variables['mdt'][:]
    # mdt = bound_arr(mdt, -2, 2)
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    print(mdt.shape)
    print(lats, lons)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection=crs)
    # ax = plt.axes(projection=crs)


    # mdt = mdt + mask

    vmin = np.min(mdt)
    vmax = np.max(mdt)
    print(vmin, vmax)

    # plt.pcolormesh(lons, lats, mdt, transform=crs)
    im = ax.pcolormesh(lons, lats, mdt, transform=crs,
                       cmap='turbo', vmin=vmin, vmax=vmax)

    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180],
                  crs=crs)
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=crs)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    # Following axes formatters cannot be used with non-rectangular projections
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.fontsize = 20
    # ax.add_feature(cfeature.LAND, resolution='10m')
    # ax.coastlines()


    # ax.gridlines()
    # ax.set_extent([100, 160, 0, 60])
    # ax.set_aspect('auto', adjustable=None)

    # cbar_arr = (np.linspace(vmin, vmax, 5, dtype=int))

    # Uncomment following to produce colorbar corresponding to each individual plot:
    fig.colorbar(im, ax=ax, fraction=0.0235, pad=0.04, ticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1., 1.5, 2],
                 format=ticker.FuncFormatter(cbar_fmt))
    plt.title('DTU10 Global Mean Dynamic Topography: 2 min resolution', fontsize= 21)
    plt.show() 






if __name__ == '__main__':
    main()

#1 min mss

# float64 lon(lon)
#     long_name: longitude
#     units: degrees_east
#     actual_range: [-1.66666667e-02  3.60016667e+02]
# unlimited dimensions:
# current shape = (21602,)
# filling on, default _FillValue of 9.969209968386869e+36 used), ('lat', <class 'netCDF4._netCDF4.Variable'>
# float64 lat(lat)
#     long_name: latitude
#     units: degrees_north
#     actual_range: [-90.  90.]
# unlimited dimensions:
# current shape = (10800,)
# filling on, default _FillValue of 9.969209968386869e+36 used), ('mss', <class 'netCDF4._netCDF4.Variable'>
# int32 mss(lat, lon)
#     long_name: mean sea surface height
#     units: m
#     scale_factor: 0.001
#     actual_range: [-105.579   86.701]
# unlimited dimensions:
# current shape = (10800, 21602)
# filling on, default _FillValue of -2147483647 used)])
# (10800, 21602)

# 2 min mss
# float64 lon(lon)
#     long_name: longitude
#     units: degrees_east
#     actual_range: [-1.66666667e-02  3.60016667e+02]
# unlimited dimensions:
# current shape = (10801,)
# filling on, default _FillValue of 9.969209968386869e+36 used), ('lat', <class 'netCDF4._netCDF4.Variable'>
# float64 lat(lat)
#     long_name: latitude
#     units: degrees_north
#     actual_range: [-90.  90.]
# unlimited dimensions:
# current shape = (5400,)
# filling on, default _FillValue of 9.969209968386869e+36 used), ('mss', <class 'netCDF4._netCDF4.Variable'>
# int32 mss(lat, lon)
#     long_name: mean sea surface height
#     units: m
#     scale_factor: 0.001
#     actual_range: [-105.578   86.694]
# unlimited dimensions:
# current shape = (5400, 10801)
# filling on, default _FillValue of -2147483647 used)])
# (5400, 10801)

# 2 min mdt
# float64 lon(lon)
#     long_name: longitude
#     units: degrees_east
#     actual_range: [-1.66666667e-02  3.60016667e+02]
# unlimited dimensions:
# current shape = (10801,)
# filling on, default _FillValue of 9.969209968386869e+36 used), ('lat', <class 'netCDF4._netCDF4.Variable'>
# float64 lat(lat)
#     long_name: latitude
#     units: degrees_north
#     actual_range: [-90.  90.]
# unlimited dimensions:
# current shape = (5400,)
# filling on, default _FillValue of 9.969209968386869e+36 used), ('mdt', <class 'netCDF4._netCDF4.Variable'>
# int16 mdt(lat, lon)
#     long_name: mean ocean dynamic topography
#     units: m
#     scale_factor: 0.001
#     actual_range: [-2.26   2.155]
# unlimited dimensions:
# current shape = (5400, 10801)
# filling on, default _FillValue of -32767 used)])
# (5400, 10801)