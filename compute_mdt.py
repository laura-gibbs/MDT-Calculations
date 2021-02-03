import math
import numpy as np
import os
import array
import matplotlib.pyplot as plt
from utils import define_dims, create_coords, bound_arr
from read_data import read_surface, read_surfaces, write_surface, apply_mask
import turbo_colormap_mpl
import matplotlib.ticker as ticker
from cartopy import config
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature



def calculate_mdt(mss, geoid, mask=True):
    mdt = mss - geoid
    if mask:
        return apply_mask(0.25, mdt)
    else:
        return mdt

def main():
    path1 = 'fortran/data/src/'
    path2 = 'fortran/data/res/'
    cmippath = '../cmip5/historical/'
    test_path = 'fortran/data/test/'
    crs = ccrs.PlateCarree(central_longitude=0)


    historical_mdts = read_surfaces('cmip5_historical_mdts_yr5.dat', cmippath, number=3,
                               start=100)
    
    example_mdt = bound_arr(historical_mdts[0].T, -1.5, 1.5)
    plt.imshow(np.rot90(example_mdt.T), cmap='turbo')
    plt.show()

    # mask_rr2 = read_surface('mask_rr0008.dat', path1)
    # plt.imshow(np.rot90(mask_rr2, 1), cmap='turbo')
    # plt.show()

    geoid = read_surface('geco_do0280_rr0004.dat', path2)
    plt.imshow(np.rot90(geoid, 1), cmap='turbo')
    plt.show()

    dtu15_geco= read_surface('dtu15_geco_do0280_rr0004.dat', test_path)
    print(np.nanmin(dtu15_geco), np.nanmax(dtu15_geco))
    dtu15_geco = bound_arr(dtu15_geco, -3, 3)
    plt.imshow(np.rot90(dtu15_geco, 1), cmap='turbo')
    plt.show()
    
    # dtu15_gtimr5 = read_surface('dtu15_go_cons_gcf_2_tim_r5_do0280_rr0004.dat', test_path)
    # print(np.nanmin(dtu15_gtimr5), np.nanmax(dtu15_gtimr5))
    # dtu15_gtimr5 = bound_arr(dtu15_gtimr5, -3, 3)
    # plt.imshow(np.rot90(dtu15_gtimr5, 1), cmap='turbo')
    # plt.show()


    # mdt = read_surface('dip_1000_dtu15gtim5do0280_rr0004.dat', path2,
                    #    transpose=True)
    
    lons, lats = create_coords(1/12)

    nemo_mdt = read_surface('orca0083_mdt_12th.dat', path2, transpose=True)
    vmin = np.nanmin(nemo_mdt)
    vmax = np.nanmax(nemo_mdt)
    print(vmin, vmax)
    # nemo_mdt = bound_arr(nemo_mdt, vmin, vmax)
    # nemo_mdt = np.rot90(nemo_mdt, 2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    im = ax.pcolormesh(lons, lats, nemo_mdt, transform=crs,
                       cmap='turbo', vmin=-1.5, vmax=1.5)   
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180], crs=crs)
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=crs)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.yaxis.set_ticks_position('both')
    ax.fontsize = 20
    ax.add_feature(cfeature.LAND)
    ax.coastlines()
    cbar = fig.colorbar(im, ax=ax, fraction=0.0235, pad=0.06, ticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    cbar.ax.set_yticklabels(['-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5'])
        # '-1.25', '-1.0', '-0.75', '-0.5', '-0.25', '0.0', '0.25', '0.5', '0.75', '1.0', '1.25'])
    cbar.ax.tick_params(axis='y', length=8, width=1, labelsize=12)
    plt.tick_params(length=10, width=1, labelright='True')
    plt.tick_params(axis='x', pad=8)
    plt.tick_params(axis='y', pad=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title('NEMO Global Mean Dynamic Topography', fontsize= 21, pad=15)
    plt.gcf().text(0.8855, 0.858, 'm', fontsize=14)
    plt.show()
    # plt.imshow(nemo_mdt, cmap='turbo')
    #plt.imshow(rcp60_mdts[0].T, cmap='turbo')
    plt.show()

    # mdt = calculate_mdt(mss, geoid)
    # calculate_mdt(apply_mask(res, mss, geoid))


if __name__ == '__main__':
    main()
