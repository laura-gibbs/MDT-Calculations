import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mdt_calculations.data_utils.utils import create_coords, get_res, bound_arr
from mdt_calculations.data_utils.dat import read_surfaces
import numpy as np
import matplotlib.colors as colors
from cartopy.feature import GSHHSFeature


def plot(arr, cmap='turbo', central_lon=0, bds=1.4, coastlines=False,
         land_feature=False, title=None, product='mdt', extent=None,
         lats=None, lons=None, low_bd=None, up_bd=None, log=False):
    arr = np.flipud(arr)
    if lats is None and lons is None:
        lons, lats = create_coords(get_res(arr), central_lon=central_lon)
    crs = ccrs.PlateCarree(central_longitude=central_lon)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    dp = '{:.1f}'
    if low_bd is not None and up_bd is not None:
        vmin = low_bd
        vmax = up_bd
        bds = None
        # ticks = np.linspace(vmin, up_bd, num=10)
    else:
        if product == 'mdt':
            vmin = -bds
            vmax = bds
        elif product == 'cs':
            vmin = 0
            vmax = bds
        elif product == 'geoid':
            vmin = -100
            vmax = 100
            cmap = 'RdBu_r'
            ticks = np.linspace(vmin, vmax, num=9)
            coastlines = True
        elif product == 'err':
            vmin = 0.01
            vmax = 0.03
            ticks = np.linspace(vmin, vmax, num=3)
            dp = '{:.2f}'
            cmap = 'gist_ncar'
            # cmap = 'nipy_spectral'
    if log:
        vmin = 0.1
        norm = colors.LogNorm(vmin, vmax)
    else:
        norm = None
    arr = bound_arr(arr, vmin, vmax)
    im = ax.pcolormesh(lons, lats, arr, transform=crs, cmap=cmap,
                       vmin=vmin, vmax=vmax, norm=norm)
    # else:
    #     im = ax.pcolormesh(lons, lats, arr, transform=crs, cmap=cmap,
    #                        vmin=vmin, vmax=vmax)        
    if extent is not None:
        if extent == 'gs':
            x0, x1 = -85, -60
            y0, y1 = 20, 45
            x_ticks = 6
            y_ticks = 6
        elif extent == 'ag':
            x0, x1 = 0, 50
            y0, y1 = -10, -50
            x_ticks = 6
            y_ticks = 6
        elif extent == 'na':
            x0, x1 = -80, -10
            y0, y1 = 20, 70
            x_ticks = 8
            y_ticks = 6
        ax.set_extent((x0, x1, y0, y1), crs=crs)
        ax.set_xticks(np.linspace(x0, x1, x_ticks), crs=crs)
        ax.set_yticks(np.linspace(y0, y1, y_ticks), crs=crs)
    else:
        ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180], crs=crs)
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=crs)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # ax.fontsize = 20
    ax.yaxis.set_ticks_position('both')
    # if land_feature:
    #     ax.add_feature(cfeature.LAND)
    if coastlines:
        ax.add_feature(GSHHSFeature(scale='intermediate', facecolor='lightgrey', linewidth=0.2))
        # ax.coastlines()
    if product == 'mdt':
        if bds == 1.5:
            ticks = np.linspace(vmin, vmax, num=7)
        elif bds == 1.4:
            ticks = np.linspace(vmin, vmax, num=15)
        elif bds == 1.25:
            ticks = np.linspace(vmin, vmax, num=11)
            dp = '{:.2f}'
        else:
            ticks = np.linspace(vmin, vmax, num=5)
            dp = '{:.2f}'
    elif product == 'cs':
        if bds == 0.5 and product == 'cs':
            ticks = np.linspace(vmin, vmax, num=6)
        else:
            ticks = np.linspace(vmin, vmax, num=6)
    if extent is None:
        labelsize = 11
        ticksize = 14
        fig.set_size_inches((20, 10.25))
        cbar = fig.colorbar(im, ax=ax, fraction=0.0235, pad=0.06, ticks=ticks)
        if product == 'mdt' or product == 'err' or product == 'geoid':
            plt.gcf().text(0.8855, 0.858, 'm', fontsize=14)
        if product == 'cs':
            plt.gcf().text(0.882, 0.858, 'm/s', fontsize=14)
    else:
        labelsize = 8
        ticksize = 9
        if extent=='gs':
            fig.set_size_inches((8, 7))
            cbar = fig.colorbar(im, ax=ax, fraction=0.041, pad=0.15, ticks=ticks)
            if product == 'mdt':
                plt.gcf().text(0.8855, 0.858, 'm', fontsize=11)
            elif product == 'cs':
                plt.gcf().text(0.869, 0.87, 'm/s', fontsize=11)
        elif extent == 'ag' or extent == 'na':
            fig.set_size_inches((9, 6))
            cbar = fig.colorbar(im, ax=ax, fraction=0.041, pad=0.15, ticks=ticks)
            # if product == 'mdt':
            #     plt.gcf().text(0.8855, 0.858, 'm', fontsize=11)
            # elif product == 'cs':
            #     plt.gcf().text(0.865, 0.89, 'm/s', fontsize=11)
    cbar.ax.set_yticklabels([dp.format(tick) for tick in ticks])
    cbar.ax.tick_params(axis='y', length=8, width=1, labelsize=7)
    plt.tick_params(length=10, width=1, labelright='True')
    plt.tick_params(axis='x', pad=8)
    plt.tick_params(axis='y', pad=3)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    if title is not None:
        plt.title(title, fontsize=21, pad=15)
    
    # plt.show()
    return fig


def plot_projection(arr, crs, vmin=0, vmax=2, cmap='turbo'):
    lons, lats = create_coords(get_res(arr), central_lon=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    arr = bound_arr(arr, vmin, vmax)
    im = ax.pcolormesh(lons, lats, arr, transform=ccrs.PlateCarree(), cmap=cmap,
                       vmin=vmin, vmax=vmax)
    fig.set_size_inches((20, 10.25))
    cbar = fig.colorbar(im, ax=ax, fraction=0.0235, pad=0.06, ticks=np.linspace(vmin, vmax, num=6))
    # cbar.ax.set_yticklabels([dp.format(tick) for tick in ticks])
    cbar.ax.tick_params(axis='y', length=8, width=1, labelsize=7)
    plt.tick_params(length=10, width=1, labelright='True')
    plt.tick_params(axis='x', pad=8)
    plt.tick_params(axis='y', pad=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig



def save_figs(dat_file, path, number, start, params, save_dir, bds=1.4):
    surfaces = read_surfaces(dat_file, path, number=number, start=start)
    params = params[start:start+number]
    for i, surface in enumerate(surfaces):
        fig = plot(surface, bds=bds, title=params[i][0]+'_'+params[i][1]+'_'+params[i][2])
        fig.set_size_inches((20, 10.25))
        fig.savefig(save_dir+params[i][0]+'_'+params[i][1]+'_'+params[i][2], dpi=300)
        plt.close()


def main():
    print("plot.py main")


if __name__ == '__main__':
    main()
