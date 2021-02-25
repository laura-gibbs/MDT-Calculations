import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from mdt_calculations.plotting.plot import plot
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mdt_calculations.data_utils.utils import create_coords, get_res, bound_arr
from mdt_calculations.data_utils.dat import read_surfaces
import numpy as np
import matplotlib.colors as colors


def multi_plot(surfaces, nrows, ncols, extent=None):#, subplot_titles=None):
    crs = ccrs.PlateCarree()
    central_lon = 0
    vmin = 0
    vmax = 2
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

    # Define the figure and each axis for the 3 rows and 3 columns
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            subplot_kw={'projection': crs},
                            figsize=(12, 8))

    for i, surface in enumerate(surfaces):
        surface = bound_arr(surface, vmin, vmax)
        lons, lats = create_coords(get_res(surface), central_lon=central_lon)
        cs = axs[i].pcolormesh(lons, lats, surface, transform=crs, cmap='turbo',
                               vmin=vmin, vmax=vmax)
        # axs[i].set_title(surface)
        axs[i].set_extent((x0, x1, y0, y1), crs=crs)
        axs[i].set_xticks(np.linspace(x0, x1, no_ticks), crs=crs)
        axs[i].set_yticks(np.linspace(y0, y1, no_ticks), crs=crs)
        lat_formatter = LatitudeFormatter()
        lon_formatter = LongitudeFormatter()
        axs[i].xaxis.set_major_formatter(lon_formatter)
        axs[i].yaxis.set_major_formatter(lat_formatter)

    # Delete the unwanted axes
    # for i in [7,8]:
    #     fig.delaxes(axs[i])
    fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.95,
                        wspace=0.2, hspace=0.5)
    cbar_ax = fig.add_axes([0.1, 0.15, 0.8, 0.025])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    # plt.suptitle('_')
    plt.show()

# def four_panel(surfaces, nrows, ncols, extent=None):#, subplot_titles=None):
#     crs = ccrs.PlateCarree()
#     central_lon = 0
#     vmin = 0
#     vmax = 2
#     if extent is not None:
#         if extent == 'gs':
#             x0, x1 = -85, -60
#             y0, y1 = 20, 45
#             no_ticks = 6
#         elif extent == 'ag':
#             x0, x1 = 0, 50
#             y0, y1 = -10, -50
#             no_ticks = 6
#         else:
#             x0, x1 = -180, 180
#             y0, y1 = -90, 90
#             no_ticks = 9

#     # Define the figure and each axis for the 3 rows and 3 columns
#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
#                             subplot_kw={'projection': crs},
#                             figsize=(12, 8))

#     for i, surface in enumerate(surfaces):
#         surface = bound_arr(surface, vmin, vmax)
#         lons, lats = create_coords(get_res(surface), central_lon=central_lon)
#         cs = axs[i].pcolormesh(lons, lats, surface, transform=crs, cmap='turbo',
#                                vmin=vmin, vmax=vmax)
#         # axs[i].set_title(surface)
#         axs[i].set_extent((x0, x1, y0, y1), crs=crs)
#         axs[i].set_xticks(np.linspace(x0, x1, no_ticks), crs=crs)
#         axs[i].set_yticks(np.linspace(y0, y1, no_ticks), crs=crs)
#         lat_formatter = LatitudeFormatter()
#         lon_formatter = LongitudeFormatter()
#         axs[i].xaxis.set_major_formatter(lon_formatter)
#         axs[i].yaxis.set_major_formatter(lat_formatter)

#     # Delete the unwanted axes
#     # for i in [7,8]:
#     #     fig.delaxes(axs[i])
#     fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.95,
#                         wspace=0.2, hspace=0.5)
#     cbar_ax = fig.add_axes([0.1, 0.15, 0.8, 0.025])
#     cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
#     # plt.suptitle('_')
#     plt.show()


def main():
    print("multiplot.py main")


if __name__ == '__main__':
    main()
