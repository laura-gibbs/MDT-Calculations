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


def main():
    crs = ccrs.PlateCarree(central_longitude=0)

    fname = ('../Aviso/mdt/MDT_CNES_CLS18_global.nc')
    dataset = netcdf_dataset(fname)


    mdt = dataset.variables['mdt'][0,:,:]
    lats = dataset.variables['latitude'][:]
    lons = dataset.variables['longitude'][:]
    print(mdt.shape)
    print(lats, lons)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection=crs)
    # ax = plt.axes(projection=crs)
    
    vmin = np.min(mdt)
    vmax = np.max(mdt)
    print(vmin, vmax)
    # mdt = bound_arr(mdt, vmin, vmax)
    vmin = np.min(mdt)
    vmax = np.max(mdt)
    print(vmin, vmax)

    # plt.pcolormesh(lons, lats, mdt, transform=crs)
    im = ax.pcolormesh(lons, lats, mdt, transform=crs,
                       cmap='turbo', vmin=-1.5, vmax=1.5)

    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180],
                  crs=crs)
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=crs)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    # Following axes formatters cannot be used with non-rectangular projections
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.yaxis.set_ticks_position('both')
    # ax.yaxis.set_major_locator()
    ax.fontsize = 20
    ax.add_feature(cfeature.LAND)
    ax.coastlines()
    # ax.gridlines()
    # ax.set_extent([100, 160, 0, 60])
    # ax.set_aspect('auto', adjustable=None)

    # cbar_arr = (np.linspace(vmin, vmax, 5, dtype=int))

    cbar = fig.colorbar(im, ax=ax, fraction=0.0235, pad=0.06, ticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    cbar.ax.set_yticklabels(['-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5'])
    cbar.ax.tick_params(axis='y', length=8, width=1, labelsize=12)
    plt.tick_params(length=10, width=1, labelright='True')
    plt.tick_params(axis='x', pad=8)
    plt.tick_params(axis='y', pad=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title('CNES-CLS18 Global Mean Dynamic Topography', fontsize=21, pad=15)
    plt.gcf().text(0.8855, 0.858, 'm', fontsize=14)
    plt.show()


if __name__ == '__main__':
    main()

    # 8th variable:
    # time, latitude, lat_bnds, longitude, long_bnds, nv, crs, mdt.

    # int32 mdt(time, latitude, longitude)
    #     _FillValue: -2147483647
    #     limitations: No data in the Mediterranean Sea
    #     coordinates: longitude latitude
    #     long_name: mean dynamic topography
    #     standard_name: mean_dynamic_topography
    #     units: m
    #     scale_factor: 0.0001
    #     grid_mapping: crs                             -coordinate ref system
    # unlimited dimensions:
    # current shape = (1, 1440, 2880)