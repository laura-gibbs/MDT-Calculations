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
    # return (count_nans(region)/(region.size) < 0.125)
    return not np.any(np.isnan(region))


def bound_arr(arr, lower_bd, upper_bd):
    arr[arr < lower_bd] = lower_bd
    arr[arr > upper_bd] = upper_bd
    return arr


def save_img(arr, name, arr_name, training=True):
    rescaled = (255.0 / np.nanmax(arr) * (arr - np.nanmin(arr))).astype(np.uint8)
    im = Image.fromarray(rescaled, mode='L')
    subdir = 'training/' if training else 'testing/'
    im.save('saved_tiles/' + subdir + 'tiles_32/' + arr_name+ '_' +name+'.png', mode='L')


def extract_overlapping_regions(arr, lat_range, tile_size=8, overlap=4):
    # lat_range is a tuple
    x_tiles = 360//(tile_size - overlap)
    y_tiles = (lat_range[1]-lat_range[0])//(tile_size - overlap)
    tile_pts = []
    for i in range(x_tiles-1):
        for j in range(y_tiles-1):
            tile_pts.append(
                (i*(tile_size - overlap), j*(tile_size - overlap) + lat_range[0])
            )
    print(tile_pts)
    regions = []
    for tile_pt in tile_pts:
        region = extract_region(arr, (tile_pt[0], tile_pt[0]+tile_size), (tile_pt[1], tile_pt[1]+tile_size))
        if valid_region(region):
            region[np.isnan(region)] = 0
            regions.append(region)
            # plt.imshow(region)
            # plt.show()
    return regions, tile_pts

cmip6_file = 'cmip6_historical_mdts_yr5_meta.txt'
cmip6_path = '../a_mdt_data/datasets/cmip6/'

cs_path = "../a_mdt_data/computations/currents/"
masks = '../a_mdt_data/computations/masks/'
mask = read_surface('mask_rr0004.dat', masks)

training_fnames = [
    'dtu18_GO_CONS_GCF_2_DIR_R5_do0280_rr0004_cs.dat',  #300
    'dtu18_GO_CONS_GCF_2_TIM_R6_do0280_rr0004_cs.dat',  #300
    'dtu18_GO_CONS_GCF_2_SPW_R5_do0280_rr0004_cs.dat',  #330
    'dtu18_GO_CONS_GCF_2_SPW_R4_do0280_rr0004_cs.dat',  #280
    'dtu18_goco06s_do0280_rr0004_cs.dat',               #300
    'dtu18_GO_CONS_GCF_2_DIR_R6_do0280_rr0004_cs.dat',  #300
    'dtu18_GO_CONS_GCF_2_TIM_R5_do0280_rr0004_cs.dat',  #280
    'dtu18_GO_CONS_GCF_2_SPW_R2_do0240_rr0004_cs.dat',  #240
    'dtu18_GTIM5_R6e_do0280_rr0004_cs.dat',             #300
    'dtu18_eigen-6c4_do0280_rr0004_cs.dat',             #2190
    'dtu18_egm2008_do0280_rr0004_cs.dat',               #2190
    'dtu18_geco_do0280_rr0004_cs.dat',                  #2190
    # 'dtu18_IGGT_R1_do0240_rr0004_cs.dat',               #240
    # 'dtu18_IfE_GOCE05s_do0250_rr0004_cs.dat',           #250
    'dtu18_GGM05c_do0280_rr0004_cs.dat',                #360
    'dtu18_GAO2012_do0280_rr0004_cs.dat',               #360
    # additional
    # 'dtu18_XGM2019e_2159_do0280_rr0004_cs.dat',         #2190
    # 'dtu18_SGG-UGM-1_do0280_rr0004_cs.dat',             #2159
    # 'dtu18_EIGEN-6C3stat_do0280_rr0004_cs.dat',        #1949
    # 'dtu18_EIGEN-6C2_do0280_rr0004_cs.dat'            #1949

]

testing_fnames = [
    'dtu18_goco05s_do0280_rr0004_cs.dat',               #280
]

for i, fname in enumerate(training_fnames):
    cs = read_surface(fname, cs_path)
    # plot(cs)
    # plt.show()
    cs = norm(bound_arr(cs + mask, 0, 2))
    # plt.imshow(cs)
    # plt.show()
    regions, tile_pts = extract_overlapping_regions(cs, (-64, 64))
    for region, tile_pt in zip(regions, tile_pts):
        save_img(region, 'tile'+str(tile_pt), str(i), training=True)

for i, fname in enumerate(testing_fnames):
    cs = read_surface(fname, cs_path)
    # plot(cs)
    # plt.show()
    cs = norm(bound_arr(cs + mask, 0, 2))
    # plt.imshow(cs)
    # plt.show()
    regions, tile_pts = extract_overlapping_regions(cs, (-64, 64))
    for region, tile_pt in zip(regions, tile_pts):
        save_img(region, 'tile'+str(tile_pt), str(i), training=False)
