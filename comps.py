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
from netCDF4 import Dataset as netcdf_dataset
from PIL import Image
import glob
import random
import math
from scipy.fft import fft2, ifft2, fftfreq, fftn, ifftn
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter


def calc_mean(txt_file, dat_file, path, fig_dir, dat_dir, mean_per='model', plot_bool=False, write=False):
    params = read_params(txt_file, path)
    models = params[:,0]
    ens = params[:,1]

    means = []
    if mean_per == "model":
        model_starts = []
        model_names = []
        for i, model in enumerate(models):
            if model not in model_names:
                model_starts.append(i)
                model_names.append(model)

        for start, end in zip(model_starts[:-1], model_starts[1:]):
            model = models[start]
            ensemble = ens[start]
            batch_size = end - start
            batch = read_surfaces(dat_file, path, number=batch_size, start=start)
            mean = np.mean(batch, axis=(0))
            means.append(mean)
            if plot_bool:
                fig = plot(mean.T)
                fig.set_size_inches((20, 10.25))
                fig.savefig(fig_dir+model+'_mean', dpi=300)
                plt.close()
            if write:
                write_surface(dat_dir+model+'_mean', mean.T)

    elif mean_per == "ensemble":
        ensemble_starts = [0]
        prev = 0
        for i, year in enumerate(params[:,2]):
            if int(year) < prev:
                ensemble_starts.append(i)
            prev = int(year)

        for start, end in zip(ensemble_starts[:-1], ensemble_starts[1:]):
            model = models[start]
            ensemble = ens[start]
            batch_size = end - start
            batch = read_surfaces(dat_file, path, number=batch_size, start=start)
            mean = np.mean(batch, axis=(0))
            means.append(mean)
            if plot_bool:
                fig = plot(mean.T)
                fig.set_size_inches((20, 10.25))
                fig.savefig(fig_dir+model+'_'+ensemble+'_mean', dpi=300)
                plt.close()
            if write:
                write_surface(dat_dir+model+'_'+ensemble+'_mean', mean.T)
    
    return means


def compute_sd(arr, fig_dir, dat_dir, prod_name, plot_bool=False, write=False):
    # Arg example: prod_name = 'cmip6_hist'
    arr = np.array(arr)
    arr[arr > 4] = np.nan
    total_mean = np.nanmean(arr, axis=(0))
    total_std = np.nanstd(arr, axis=(0))
    total_mean = total_mean.T
    total_std = total_std.T
    if plot_bool:
        fig = plot(total_mean)#, low_bd=.35, up_bd=0.8)
        fig.set_size_inches((20, 10.25))
        fig.savefig(fig_dir+prod_name+'_mean', dpi=300)
        plt.close()
        fig = plot(total_std)#, low_bd=.35, up_bd=0.8)
        fig.set_size_inches((20, 10.25))
        fig.savefig(fig_dir+prod_name+'_std', dpi=300)
        plt.close()
    if write:
        write_surface(dat_dir+prod_name+'_mean', total_mean)
        write_surface(dat_dir+prod_name+'_std', total_std)
    
    return total_mean, total_std


def pad_rgba(im, pad):
    return np.concatenate(
        (
            np.pad(im[:, :, :3], ((pad, pad), (pad, pad), (0, 0)), constant_values=0),
            np.pad(im[:, :, 3:], ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
        ),
        axis=-1
    )


def savegrid(ims, rows=12, cols=12, pad=1, shuffle=True):
    # get number of images and the square root (corresponding to grid size, e.g. 144 > 12 x 12)
    N = len(ims)
    Nw = int(math.sqrt(N))
    
    if shuffle:
        random.shuffle(ims)
    ims = np.array([np.array(im) for im in ims])
    # ims = np.array([pad_rgba(ims[i], pad) for i in range(N)])
    size = ims.shape[1]
    # big_im = np.zeros((size * Nw, size * Nw, 4), dtype=np.uint8)
    # for i in range(Nw):
    #     for j in range(Nw):
    #         big_im[i * size: (i + 1) * size, j * size: (j + 1) * size] = ims[i * Nw + j]
    # big_im = pad_rgba(big_im, 1)


    big_im = np.zeros(((size + pad) * Nw + pad, (size + pad) * Nw + pad, 4), dtype=np.uint8)
    big_im[:, :, 3] = 255
    for i in range(Nw):
        for j in range(Nw):
            big_im[
                pad + i * (size + pad): (i + 1) * (size + pad),
                pad + j * (size + pad): (j + 1) * (size + pad)
            ] = ims[i * Nw + j]

    ax = plt.subplot(111)
    ax.set_axis_off()
    ax.imshow(big_im)
    plt.show()
    return big_im


def easy_plot(dat_file, dat_path, product, bds, figs_dir, figname, log=True):
    surface = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
    fig = plot(surface, bds=bds, product=product)
    fig.savefig(figs_dir+figname+'_cs', dpi=300)
    plt.close()
    surface = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
    fig = plot(surface, bds=bds, product=product, extent='gs')
    fig.savefig(figs_dir+figname+'_gs', dpi=300)
    plt.close()
    surface = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
    fig = plot(surface, bds=bds, product=product, extent='ag')
    fig.savefig(figs_dir+figname+'_ag', dpi=300)
    plt.close()
    if log:
        surface = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
        fig = plot(surface, bds=bds, product=product, log=True)
        fig.savefig(figs_dir+figname+'_cs_log', dpi=300)
        plt.close()
        surface = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
        fig = plot(surface, bds=bds, product=product, extent='gs', log=True)
        fig.savefig(figs_dir+figname+'_gs_log', dpi=300)
        plt.close()
        surface = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
        fig = plot(surface, bds=bds, product=product, extent='ag', log=True)
        fig.savefig(figs_dir+figname+'_ag_log', dpi=300)
        plt.close()
    return surface


def extract_region(mdt, lon_range, lat_range, central_lon=0, central_lat=0):
    res = mdt.shape[0] // 180

    px = ((lon_range[0] + central_lon) * res, (lon_range[1] + central_lon) * res)
    py = ((-lat_range[1] + 90) * res, (-lat_range[0] + 90) * res)
    print('here', py)
    return mdt[py[0]:py[1], px[0]:px[1]]


def norm(a):
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))


def bound_arr(arr, lower_bd, upper_bd):
    arr[arr < lower_bd] = lower_bd
    arr[arr > upper_bd] = upper_bd
    return arr


def apply_gaussian(arr, sigma=1.6, bd=-1.5):
    V = arr.copy()
    mask = np.ones_like(arr)
    mask[np.isnan(arr)] = np.nan
    V[np.isnan(arr)] = 0
    VV = gaussian_filter(V, sigma=sigma)

    W = 0*arr.copy() + 1
    W[np.isnan(arr)] = 0
    WW = gaussian_filter(W, sigma=sigma)

    arr = VV/WW * mask
    # arr[np.isnan(arr)] = bd

    return arr


def main():
    mdts = '../a_mdt_data/computations/mdts/'
    mss = '../a_mdt_data/computations/mss/'
    geoids = '../a_mdt_data/computations/geoids/'
    cs = '../a_mdt_data/computations/currents/'
    masks = '../a_mdt_data/computations/masks/'
    figs_dir = './figs/'
    mask = read_surface('mask_rr0004.dat', '../a_mdt_data/computations/masks/')
    cmip5_path = '../a_mdt_data/datasets/cmip5/historical/'
    rcp60 = '../a_mdt_data/datasets/cmip5/rcp60/' 
    cmip5_file = 'cmip5_historical_mdts_yr5_meta.txt'
    cmip5_datfile = 'cmip5_historical_mdts_yr5.dat'

    cmip6_file = 'cmip6_historical_mdts_yr5_meta.txt'
    cmip6_path = '../a_mdt_data/datasets/cmip6/'
    cmip6_datfile = 'cmip6_historical_mdts_yr5.dat'
    cmip5_models = '../a_mdt_data/computations/cmip5_calcs/model_means/'
    cmip6_models = '../a_mdt_data/computations/cmip6_calcs/model_means/'

    # dtu_path = '../a_mdt_data/datasets/dtu/'
    # cmip5_hist = read_surfaces('cmip5_historical_mdts_yr5.dat', cmip5_path, number=31, start=4707)
    # mean_mdt = np.nanmean(cmip6_hist, axis=(0))
    # fig = plot(mean_mdt, bds=3)
    # fig.set_size_inches((20, 10.25))

    nemo_test = read_surface(r"C:\Users\oa18724\Documents\Master_PhD_folder\a_mdt_data\HR_model_data\nemo_currents\ORCA0083-N06_1958m02_ssha_12th_cs.dat")
    plot(nemo_test, product='cs')
    plt.show()

    test = read_surface('../a_mdt_data/HR_model_data/cmip6_models/CMCC-CM2-HR4_r1i1p1f1_gn_1851.dat')
    print('test:', np.any(np.isnan(test)))
    # plot(test, product='cs')
    # plt.show()

    # year = 1851
    # for i in range(31):
    #     filename = 'cmip5_models/MPI-ESM-MR_r3i1p1_' + str(year)
    #     write_surface(filename, cmip5_hist[i])
    #     year = year + 5

    # means = calc_mean(cmip6_file, cmip6_datfile, cmip6_path, '../a_mdt_data/figs/cmip6/model_means/', '../a_mdt_data/computations/cmip6_calcs/model_means/',
    #                   mean_per='model')
    # total_mean, total_std = compute_std(means, '../a_mdt_data/figs/cmip6/', '../a_mdt_data/computations/cmip6_calcs/', 'cmip6_hist')

    # # fig.savefig(figs_dir+'cls/cls18_cs', dpi=300)

    target_img = np.load('../a_mdt_data/HR_model_data/training_regions/CNRM-CM6-1-HR_r1i1p1f2_gn_2001_cs_0_488.npy')
    quilt = Image.open('../MDT-Denoising/quilting/DCGAN_quilted/cut-b32-n5_0.png')
    quilt = quilt.convert(mode='L')
    quilt = np.array(quilt).astype(np.float32)
    mask = target_img != 0
    quilt = (quilt - np.nanmin(quilt)) / (np.nanmax(quilt) - np.nanmin(quilt))
    img = target_img + .3* quilt * mask
    img = apply_gaussian(img) * mask
    plt.imshow((target_img), cmap='turbo')
    plt.show()

    # plot(cs_surface, product='cs', bds=1.4, extent='test')#, lats=(32,64), lons=(32, 64))

    mpi_cs = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
    # mpi_cs = np.flipud(mpi_cs)
    mpi_cs = extract_region(mpi_cs, (-85, -55), (15, 45))
    print(np.nanmax(mpi_cs))
    mpi_cs[mpi_cs > 3.5] = 0
    print(np.nanmax(mpi_cs))
    mpi_cs = norm(bound_arr(mpi_cs, 0, 2))
    
    # img_src = Image.open('quilting/DCGAN_quilted/cut-b32-n5_0.png').convert('L')
    # img_src = Image.open('quilting/WAE_MMD2_32_quilted/cut-b32-n5_0.png').convert('L')
    # noise = np.array(img_src)

    # noise = .4* norm(noise)
    # noise = noise[0:120, 0:120]
    # mpi_noise = mpi_cs + noise
    # mpi_noise[np.isnan(mpi_noise)] = 0
    # mpi_cs[np.isnan(mpi_cs)] = 0
    # fig, ax  = plt.subplots(2, 2)
    # ax[0][0].imshow(mpi_cs, cmap='turbo')
    # ax[0][1].imshow(noise, cmap='turbo')
    # ax[1][0].imshow(mpi_noise, cmap='turbo')
    # ax[1][1].imshow(cs_surface, cmap='turbo')
    # plt.show()


    # cm = plt.get_cmap('turbo')
    # img_src = Image.open('figs/meeting/390.png').convert('L')
    # img = np.array(img_src)
    # img = cm(img)
    # img = np.uint8(img * 255)
    # img = Image.fromarray(img)
    # img.save('figs/meeting/390_turbo.png')

    # # Grab 244 
    # img_paths = glob.glob('../PyTorch-VAE/savedtiles/wae_mmd_rbf/WAE_MMD1_s64_0/*.png')
    # imgs = []
    
    # for img_path in img_paths:
    #     img = Image.open(img_path).convert('L')
    #     img = np.array(img)
    #     img = cm(img)
    #     img = np.uint8(img * 255)
    #     img = Image.fromarray(img)
    #     imgs.append(img)
    #     print('imgs:', len(imgs))
    # imgs = imgs[:144]
    # print('imgs:', len(imgs))
    # savegrid(imgs, shuffle=False)


if __name__ == '__main__':
    main()