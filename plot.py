import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import turbo_colormap_mpl
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import read_data as rd
from scipy.ndimage import gaussian_filter
from PIL import Image
from matplotlib import cm
from utils import define_dims, create_coords


def cbar_fmt(x, pos):
    r"""Colorbar function, formats colorbar labels to include units 'm'.

    Args:
        x (float): colorbar label value
        pos (int): position of colorbar label

    Returns:
        str(x) + 'm' (String)

    """
    return str(x) + 'm/s'


def plot_surface(surface, ax, fig, res, coast, mdt_bool, sptitle):
    r"""Plots input surface.

    Args: surface


    Returns:

    """
    print("Res=", res)
    lon, lat = create_coords(res)
    # print(lon.shape, lat.shape)
    crs_latlon = ccrs.PlateCarree(central_longitude=0)

    if mdt_bool:
        vmin = -1.5
        vmax = 1.5
        cbar_arr = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    else:
        vmin = 0
        vmax = 0.5
        # vmin = np.min(surface)
        # vmax = np.max(surface)
        print(vmin, vmax)
    # ax.set_extent((-80.0, 20.0, 10.0, 80.0), crs=crs_latlon)
    # ax.set_xlabel('xlabel', fontsize=10)
    print("Number of lon/lat elements", len(lon), len(lat))
    im = ax.pcolormesh(lon, lat, surface, transform=crs_latlon,
                       cmap='turbo', vmin=vmin, vmax=vmax)

    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180],
                  crs=crs_latlon)
    
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=crs_latlon)

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    # Following axes formatters cannot be used with non-rectangular projections
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.fontsize = 20

    if coast:
        ax.coastlines(resolution='110m', linewidth=1.2)  # 10m, 50m or 110m

    cbar_arr = (np.linspace(vmin, vmax, 5, dtype=int))

    # Uncomment following to produce colorbar corresponding to each individual plot:
    fig.colorbar(im, ax=ax, fraction=0.0235, pad=0.04, ticks=[-1.5, -1, -0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],#, 3.5, 4],
                 format=ticker.FuncFormatter(cbar_fmt))
                #fraction=0.0235 for vertical cbar
    return im, vmin, vmax


def multi_plot(surfaces, crs_latlon, coasts, mdt_bools, subplot_titles):
    r"""Plots multiple surfaces in one plot.
        Dynamically plots up to 6 surfaces.
        Uses plot_surface for each input surface.

    Args:
        surfaces (list): list of numpy arrays containing surfaces.
        coasts (list): list of booleans corresponding to coastlines.
        mdt_bools (list): list of booleans; True if MDT.
        subplot_titles (list): list of subplot titles.

    Returns:
        : Plot containing subplots of input surfaces.
    """
    
    print(f'len(surfaces)={len(surfaces)}')
    if len(surfaces) == 1:
        fig = plt.figure(figsize=(10, 5))
        axs = [
                fig.add_subplot(1, 1, 1, projection=crs_latlon)
        ]
    elif len(surfaces) == 2:
        fig = plt.figure(figsize=(10, 5))
        axs = [
                fig.add_subplot(1, 2, 1, projection=crs_latlon),
                fig.add_subplot(1, 2, 2, projection=crs_latlon)
        ]
    elif len(surfaces) == 3:
        fig = plt.figure(figsize=(10, 5))
        axs = [
                fig.add_subplot(2, 2, 1, projection=crs_latlon),
                fig.add_subplot(2, 2, 2, projection=crs_latlon),
                fig.add_subplot(2, 2, 3, projection=crs_latlon)
        ]
    elif len(surfaces) == 4:
        fig = plt.figure(figsize=(10, 5))
        axs = [
                fig.add_subplot(2, 2, 1, projection=crs_latlon),
                fig.add_subplot(2, 2, 2, projection=crs_latlon),
                fig.add_subplot(2, 2, 3, projection=crs_latlon),
                fig.add_subplot(2, 2, 4, projection=crs_latlon)
        ]
    elif len(surfaces) == 5:
        fig = plt.figure(figsize=(10, 5))
        axs = [
                fig.add_subplot(2, 3, 1, projection=crs_latlon),
                fig.add_subplot(2, 3, 2, projection=crs_latlon),
                fig.add_subplot(2, 3, 3, projection=crs_latlon),
                fig.add_subplot(2, 3, 4, projection=crs_latlon),
                fig.add_subplot(2, 3, 5, projection=crs_latlon)
        ]
    elif len(surfaces) == 6:
        fig = plt.figure(figsize=(10, 5))
        axs = [
                fig.add_subplot(2, 3, 1, projection=crs_latlon),
                fig.add_subplot(2, 3, 2, projection=crs_latlon),
                fig.add_subplot(2, 3, 3, projection=crs_latlon),
                fig.add_subplot(2, 3, 4, projection=crs_latlon),
                fig.add_subplot(2, 3, 5, projection=crs_latlon),
                fig.add_subplot(2, 3, 6, projection=crs_latlon)
        ]

    ims = []
    i = 1
    # print(len(surfaces), len(axs), len(coasts), len(mdt_bools), len(subplot_titles))
    for surface, ax, coast, mdt_bool, sptitle in zip(surfaces, axs, coasts,
                                                     mdt_bools,
                                                     subplot_titles):
        print(f"running loop number: {i}")
        i = i + 1
        res = 360/surface.shape[1]
        print("surface.shape =", surface.shape)
        print(res)
        im, vmin, vmax = plot_surface(surface, ax, fig, res, coast, mdt_bool,
                                      sptitle)
        ims.append(im)

    cbar_arr = (np.linspace(vmin, vmax, 8, dtype=int))

    # Uncomment following for single colorbar on multiplot:
    # fig.colorbar(ims[0], ax=axs, fraction=0.025, pad=0.04, ticks=cbar_arr,
                #  format=ticker.FuncFormatter(cbar_fmt))
    
    # fig.suptitle("The MDT calculated from the DTU15 MSS and TIM_R5 Geoid", fontsize=14, weight='bold', y=0.93)
    plt.tight_layout()
    
    # plt.savefig('./Experiments/fig_dpi10.eps', format='eps', dpi=10)
    plt.show()
    

def save_img(img):
    img[np.isnan(img)] = np.nanmin(img)
    # img = np.clip(img, -1.5, 1.5)
    # img = np.arcsinh(img)
    img = (img-np.min(img))/(np.max(img)-np.min(img))
    img = Image.fromarray((img[:, :] * 255).astype(np.uint8))
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save('gmdt_img.png')


def grey_to_colour(img):
    cm = plt.get_cmap('turbo')
    print(type(img))
    colored_img = np.clip(img, -1.5, 1.5)
    colored_img = (colored_img-np.min(colored_img))/(np.max(colored_img)-np.min(colored_img))
    print(np.amax(colored_img))
    colored_img = cm(colored_img)
    print(colored_img.shape, np.mean(colored_img[:, :, 3]))
    colored_img = Image.fromarray((colored_img[:, :, :3] * 255).astype(np.uint8))
    colored_img = colored_img.transpose(Image.FLIP_TOP_BOTTOM)
    return colored_img


def main():
    fname = "./data/res/dtu15_gtim5_do0280_rr0004.dat"
    proj = ccrs.PlateCarree(central_longitude=180)
    mss = rd.read_dat(fname)
    print(mss.shape)
    # V=mss.copy()
    # mask = np.ones_like(mss)
    # mask[np.isnan(mss)] = np.nan
    # V[np.isnan(mss)]=0
    # VV=gaussian_filter(V,sigma=3)

    # W=0*mss.copy()+1
    # W[np.isnan(mss)]=0
    # WW=gaussian_filter(W,sigma=3)

    # rd.write_dat(mss, './data/res/gaussian030_mdt_do0280_rr0004.dat', overwrite=True)

    # mss=VV/WW * mask
    # rd.write_dat(mss, './data/res/gaussian030_mdt_do0280_rr0004.dat', overwrite=True)
    # mss[np.isnan(mss)] = -1.5

    f_n = './data/res/dip_smooth0mask_3500_dtu15gtim5do0280_rr0004.dat'
    gmdt = rd.read_dat(f_n)
    multi_plot([gmdt], proj, [False], [True], [""])

    current_proj = ccrs.PlateCarree(central_longitude=180)
    f = "./fortran/data/dip_smooth0mask_2000_cs.dat"
    currents = rd.read_dat(f, res=0.25)

    multi_plot([currents], current_proj, [False], [False], [""])
    # img = save_img(gmdt)
    # print(img.size)


if __name__ == '__main__':
    main()
