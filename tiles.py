import cartopy.crs as ccrs
from mdt_calculations.data_utils.netcdf import load_cls, load_dtu  # load_dtu - pc breaks with big dtu mdts
from mdt_calculations.plotting.plot import plot, save_figs, plot_projection
from mdt_calculations.plotting.multiplot import multi_plot
from mdt_calculations.data_utils.dat import read_surface, read_surfaces, write_surface, read_params
from mdt_calculations.computations.wrapper import mdt_wrapper, cs_wrapper
from mdt_calculations.plotting.gifmaker import gen_gif
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


def extract_region(mdt, lon_range, lat_range, central_lon=0, central_lat=0):
    res = mdt.shape[0] // 180

    px = ((lon_range[0] + central_lon) * res, (lon_range[1] + central_lon) * res)
    py = ((lat_range[0] + 90) * res, (lat_range[1] + 90) * res)

    return mdt[py[0]:py[1], px[0]:px[1]]


def norm(a):
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))


def count_nans(region):
    return np.count_nonzero(np.isnan(region))


def valid_region(region):
    print('region size =', region.size)
    return (count_nans(region)/(region.size) < 0.25)


def bound_arr(arr, lower_bd, upper_bd):
    arr[arr < lower_bd] = lower_bd
    arr[arr > upper_bd] = upper_bd
    return arr


def save_img(arr, name):
    rescaled = (255.0 / np.nanmax(arr) * (arr - np.nanmin(arr))).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save('saved_tiles/'+name+'.png')


def extract_overlapping_regions(mdt, lat_range, tile_size=10, overlap=5):
    # lat_range is a tuple
    x_tiles = 360//(tile_size - overlap)
    y_tiles = (lat_range[1]-lat_range[0])//(tile_size - overlap)
    tile_pts = []
    for i in range(x_tiles):
        for j in range(y_tiles):
            tile_pts.append(
                (i*(tile_size - overlap), j*(tile_size - overlap) + lat_range[0])
            )
    print(tile_pts)
    regions = []
    for tile_pt in tile_pts:
        region = extract_region(mdt, (tile_pt[0], tile_pt[0]+tile_size), (tile_pt[1], tile_pt[1]+tile_size))
        if valid_region(region):
            regions.append(region)
            save_img(region, 'tile_'+str(tile_pt))
            plt.imshow(region)
            plt.show()
    return regions


cs_path = "../a_mdt_data/computations/currents/"
masks = '../a_mdt_data/computations/masks/'
mask = read_surface('mask_rr0004.dat', masks)
cs = read_surface('dtu18_gtim5_do0280_rr0004_cs_band20.dat', cs_path)
cs = norm(bound_arr(cs + mask, 0, 2))
plt.imshow(cs)
plt.show()
extract_overlapping_regions(cs, (-55, 55))
