import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mdt_calculations.data_utils.utils import create_coords, get_res, bound_arr
import cartopy.feature as cfeature
import numpy as np


def plot(arr, cmap='turbo', central_lon=0, bds=1.4, coastlines=False,
         land_feature=False, title=None, product='mdt', extent=None):
    lons, lats = create_coords(get_res(arr), central_lon=central_lon)
    crs = ccrs.PlateCarree(central_longitude=central_lon)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    if product == 'mdt':
        vmin = -bds
        vmax = bds
    if product == 'cs':
        vmin = 0
        vmax = bds
    arr = bound_arr(arr, vmin, vmax)
    im = ax.pcolormesh(lons, lats, arr, transform=crs,
                       cmap=cmap, vmin=vmin, vmax=vmax)   
    if extent == 'gs':
        ax.set_extent((95, 120, 20, 45), crs=crs)
        ax.set_xticks(np.linspace(95, 120, 6), crs=crs)
        ax.set_yticks(np.linspace(20, 45, 6), crs=crs)
    else:
        ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180], crs=crs)
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=crs)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.yaxis.set_ticks_position('both')
    ax.fontsize = 20
    if land_feature:
        ax.add_feature(cfeature.LAND)
    if coastlines:
        ax.coastlines()
    dp = '{:.1f}'
    if product == 'mdt':
        if bds == 1.5:
            ticks = np.linspace(vmin, vmax, num=7)
        elif bds == 1.4:
            ticks = np.linspace(vmin, vmax, num=15)
        elif bds == 1.25:
            ticks = np.linspace(vmin, vmax, num=11)
            dp = '{:.2f}'
        else:
            ticks = np.linspace(vmin, vmax)
    if product == 'cs':
        if bds == 0.5 and product == 'cs':
            ticks = np.linspace(vmin, vmax, num=6)
        else:
            ticks = np.linspace(vmin, vmax, num=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.0235, pad=0.06, ticks=ticks)
    cbar.ax.set_yticklabels([dp.format(tick) for tick in ticks])
    cbar.ax.tick_params(axis='y', length=8, width=1, labelsize=12)
    plt.tick_params(length=10, width=1, labelright='True')
    plt.tick_params(axis='x', pad=8)
    plt.tick_params(axis='y', pad=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if title is not None:
        plt.title(title, fontsize=21, pad=15)
    if product == 'mdt':
        plt.gcf().text(0.8855, 0.858, 'm', fontsize=14)
    if product == 'cs':
        plt.gcf().text(0.882, 0.858, 'm/s', fontsize=14)
    plt.show()


def main():
    print("plot.py main")


if __name__ == '__main__':
    main()
