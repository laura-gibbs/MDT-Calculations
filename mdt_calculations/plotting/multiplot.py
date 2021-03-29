import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from mdt_calculations.plotting.plot import plot
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mdt_calculations.data_utils.utils import create_coords, get_res, bound_arr
from mdt_calculations.data_utils.dat import read_surfaces
import numpy as np
import matplotlib.colors as colors


def multi_plot(surfaces, product='mdt', extent=None, axtitles=None,
               coastlines=False, stacked=True):  # , subplot_titles=None):
    crs = ccrs.PlateCarree()
    panel = len(surfaces)
    central_lon = 0
    cbarwidth = 0.02
    y_ticks = 7
    if product == 'mdt':
        vmin = -1.4
        vmax = 1.4
        cmap = 'turbo'
        cticks_no = 15
    elif product == 'cs':
        vmin = 0
        vmax = 2
        cmap = 'turbo'
        cticks_no = 9
    elif product == 'geoid':
        vmin = -100
        vmax = 100
        cmap = 'RdBu_r'
        # cmap = 'jet'
        cticks_no = 9
    elif product == 'err':
        vmin = 0
        vmax = 0.03
        ticks = np.linspace(vmin, vmax, num=4)
        dp = '{:.2f}'
        cmap = 'gist_ncar'
    if extent is not None:
        if extent == 'gs':
            x0, x1 = -85, -60
            y0, y1 = 20, 45
            no_ticks = 6
        elif extent == 'ag':
            x0, x1 = 0, 50
            y0, y1 = -10, -50
            no_ticks = 6
    else:
        x0, x1 = -180, 180
        y0, y1 = -90, 90
        no_ticks = 9
        y_ticks = 7
    if panel == 2:
        if extent is None:
            if stacked:
                nrows, ncols = 2, 1
                figsize = (12, 12.8)
                wspace, hspace = 0.23, 0.13
                bottom, top = 0.09, 0.96
            else:
                cbarwidth = 0.03
                nrows, ncols = 1, 2
                figsize = (19, 5.75)
                wspace, hspace = 0.08, 0.02
                bottom, top = 0.04, 0.98
        else:
            nrows, ncols = 1, 2
            figsize = (12, 8)
            wspace, hspace = 0.2, 0.5
            bottom, top = 0.25, 0.9
    elif panel == 4:
        nrows, ncols = 2, 2
        figsize = 20, 10.5
        wspace, hspace = 0.18, 0.14
        bottom, top = 0.1, 0.98
        if extent is not None:
            figsize = (11, 11)
            if extent == 'ag':
                figsize = (11, 9.1)
    cticks = np.linspace(vmin, vmax, num=cticks_no)
    # Define the figure and each axis for the 3 rows and 3 columns
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            subplot_kw={'projection': crs},
                            figsize=figsize)
    axs = axs.flatten()

    for i, surface in enumerate(surfaces):
        surface = bound_arr(surface, vmin, vmax)
        lons, lats = create_coords(get_res(surface), central_lon=central_lon)
        cs = axs[i].pcolormesh(lons, lats, surface, transform=crs, cmap=cmap,
                               vmin=vmin, vmax=vmax)
        if axtitles is not None:
            axs[i].set_title(axtitles[i])
        axs[i].set_extent((x0, x1, y0, y1), crs=crs)
        axs[i].set_xticks(np.linspace(x0, x1, no_ticks), crs=crs)
        axs[i].set_yticks(np.linspace(y0, y1, y_ticks), crs=crs)
        lat_formatter = LatitudeFormatter()
        lon_formatter = LongitudeFormatter()
        axs[i].xaxis.set_major_formatter(lon_formatter)
        axs[i].yaxis.set_major_formatter(lat_formatter)
        if coastlines:
            axs[i].coastlines(linewidth=0.5)
    # Delete the unwanted axes
    # for i in [7,8]:
    #     fig.delaxes(axs[i])
    fig.subplots_adjust(bottom=bottom, top=top, left=0.05, right=0.95,
                        wspace=wspace, hspace=hspace)
    # cbar_ax = fig.add_axes([0.1, 0.04, 0.8, 0.02])
    cbar_ax = fig.add_axes([0.1, 0.04, 0.8, cbarwidth])
    cbar = fig.colorbar(cs, cax=cbar_ax, ticks=cticks, orientation='horizontal')
    # plt.suptitle('GOCE GTIM5 Geoid')
    plt.show()


def main():
    print("multiplot.py main")


if __name__ == '__main__':
    main()
