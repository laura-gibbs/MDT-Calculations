from scipy.ndimage import gaussian_filter
from mdt_calculations.plotting.plot import plot, save_figs, plot_projection
from mdt_calculations.plotting.multiplot import multi_plot
from mdt_calculations.data_utils.dat import read_surface, read_surfaces, write_surface, read_params
import numpy as np
import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
from skimage.transform import rescale, resize
from skimage.restoration import inpaint
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.metrics import structural_similarity as ssim


EXTENTS = {
    'na': ((-80, -10), (20, 70)),
    'ag': ((0, 50), (-10, 50)),
}


def norm(a):
    return (a - a.min()) / (a.max() - a.min())


def extract_region(mdt, lon_range, lat_range, central_lon=0, central_lat=0):
    res = mdt.shape[0] // 180

    px = ((lon_range[0] + central_lon) * res, (lon_range[1] + central_lon) * res)
    py = ((lat_range[0] + 90) * res, (lat_range[1] + 90) * res)

    return mdt[py[0]:py[1], px[0]:px[1]]


def pmf(arr):
    pmf_arr = []
    iterations = []
    for i in range(350):
        print(f"running iteration {i}")
        mdt = anisotropic_diffusion(mdt, niter=1, kappa=.16, gamma=0.01, option=2)
        pmf_mdts.append(mdt)
        iterations.append(i)
    return pmf_arr, iterations

def main():
    mdts = '../a_mdt_data/computations/mdts/'
    gauss_filtered = '../a_mdt_data/computations/mdts/gauss_filtered/'
    cs = '../a_mdt_data/computations/currents/'
    gauss_cs_path = '../a_mdt_data/computations/currents/gauss_filtered/'
    masks = '../a_mdt_data/computations/masks/'
    cmip6_models = '../a_mdt_data/computations/cmip6_calcs/model_means/'
    mask = read_surface('mask_rr0008.dat', masks)
    mask_4 = read_surface('mask_rr0004.dat', masks)
    mask_12 = read_surface('land_mask_12thdeg.dat', masks)
    fig_dir = 'figs/mdt_plots/dtu18_gtim5_filtered/'

    dtu18_gtim5 = read_surface('sh_mdt_DTU18_GTIM5_L280.dat', mdts)
    # dtu18_gtim5[np.isnan(dtu18_gtim5)] = 0
    # dtu18_gtim5 = np.clip(dtu18_gtim5, -1.4, 1.4)
    # dtu18_gtim5 = norm(dtu18_gtim5)

    # Loading filtered currents
    gauss_cs = read_surface('dtu18_gtim5_gauss_250km_cs.dat', cs)
    gauss_cs[np.isnan(gauss_cs)] = 0
    pmf_cs = read_surface('dtu18_gtim5_pmf_350i_16k_01g_cs.dat', cs)#
    pmf_cs[np.isnan(pmf_cs)] = 0

    # plot(gauss_cs, product='cs')
    # plot(pmf_cs, product='cs')
    # plt.show()

    cls_mdt = read_surface('cls18_cs.dat', cs)
    cls_mdt[np.isnan(cls_mdt)] = 0
    cls_downscaled = rescale(cls_mdt, 0.5)
    # cls_downscaled = np.clip(cls_downscaled, -1.4, 1.4)
    # cls_downscaled = norm(cls_downscaled)
    
    nemo_mdt = read_surface('orca0083_cs.dat', cs)
    nemo_mdt[np.isnan(nemo_mdt)] = 0
    nemo_downscaled = resize(nemo_mdt, (720, 1440))
    # nemo_downscaled = np.clip(nemo_downscaled, -1.4, 1.4)
    # nemo_downscaled = norm(nemo_downscaled)

    # Gaussian filter
    filter_widths = []
    gauss_mdts = []
    for i in range(32):
        k = (i+1)/4
        mdt = gaussian_filter(dtu18_gtim5, sigma=k) + mask_4
        mdt[np.isnan(mdt)] = 0
        gauss_mdts.append(mdt)
        r = int((int((4*k)+0.5))/4 * 111)
        filter_widths.append(r)
        # write_surface(f'dtu18_gtim5_gauss_{r}km', gauss_mdts[i], gauss_filtered)
    

    # Plot Gaussian filter widths: for r = 5, 9, 13, 17
    gauss_plots = []
    for i in range(4):
        k = ((i+1)*4) 
        image = resize(gauss_mdts[k], (1440, 2880), order=3) + mask
        gauss_plots.append(image)
        # plot(image, product='mdt', extent='na')
        # plt.show()
        # plt.close()

    # Load gaussian currents
    gauss_cs = []
    for i in range(32):
        k = (i+1)/4
        r = int((int((4*k)+0.5))/4 * 111)
        cs = read_surface(f'dtu18_gtim5_gauss_{r}km_cs.dat', gauss_cs_path)
        cs[np.isnan(cs)] = 0
        gauss_cs.append(cs)
        # if i % 8:
        #     plot(cs, product='cs', central_lon=180)
        #     plt.show()
        #     plt.close()



    # PMF Filter
    # pmf_mdts = []
    # iterations = []
    # mdt = dtu18_gtim5
    # for i in range(350):
    #     print(f"running iteration {i}")
    #     mdt = anisotropic_diffusion(mdt, niter=1, kappa=.16, gamma=0.01, option=2)
    #     pmf_mdts.append(mdt)
    #     iterations.append(i)

    # extent = 'na'
    # plot(resize(dtu18_gtim5, (1440, 2880), order=3) + mask, product='mdt', extent=extent)
    # plt.show()

    # # Gaussian mdt filtered 200km radius
    # image = resize(gauss_mdts[6], (1440, 2880), order=3) + mask
    # plot(image, product='mdt', extent=extent)
    # plt.show()

    # # Gaussian: Residual from original mdt
    # gauss_residual = dtu18_gtim5 - gauss_mdts[6]
    # image = resize(gauss_residual, (1440, 2880), order=3) + mask
    # plot(image, product='mdt', extent=extent, low_bd=-0.15, up_bd=0.15)
    # plt.show()

    # # PMF filtered mdt 350 iterations
    # image = resize(pmf_mdts[349], (1440, 2880), order=3) + mask
    # plot(image, product='mdt', extent=extent)
    # plt.show()

    # # PMF: Residual from original mdt
    # pmf_residual = dtu18_gtim5 - pmf_mdts[349]
    # image = resize(pmf_residual, (1440, 2880), order=3) + mask
    # plot(image, product='mdt', extent=extent, low_bd=-0.15, up_bd=0.15)
    # plt.show()
    
    # # PMF - gaussian MDT residual
    # pmf_gauss = pmf_mdts[349] - gauss_mdts[6]
    # image = resize(pmf_gauss, (1440, 2880), order=3) + mask
    # plot(image, product='mdt', extent=extent, low_bd=-0.1, up_bd=0.1)
    # plt.show()

    lon_range, lat_range = EXTENTS['na']
    gauss_cs = [extract_region(mdt, lon_range, lat_range) for mdt in gauss_cs]
    cls_downscaled = extract_region(cls_downscaled, lon_range, lat_range)
    nemo_downscaled = extract_region(nemo_downscaled, lon_range, lat_range)

    # RMSE
    rmse_cls = []
    ss_cls = []
    for mdt in gauss_cs:
        # print(cs.shape, cls_downscaled.shape)
        rmse = np.sqrt(np.nanmean(((mdt - cls_downscaled)**2)))
        rmse_cls.append(rmse)
        ss = ssim(mdt, cls_downscaled)
        ss_cls.append(ss)
    rmse_cls = np.array(rmse_cls)
    rmse_cls = norm(rmse_cls) #+ 0.01
    ss_cls = np.array(ss_cls)

    rmse_nemo = []
    ss_nemo = []
    for mdt in gauss_cs:
        rmse = np.sqrt(np.nanmean(((mdt - nemo_downscaled)**2)))
        rmse_nemo.append(rmse)
        ss = ssim(mdt, nemo_downscaled)
        ss_nemo.append(ss)
    rmse_nemo = np.array(rmse_nemo)
    rmse_nemo = norm(rmse_nemo)# + 0.01
    ss_nemo = np.array(ss_nemo)
    
    rmse_summed = rmse_cls + rmse_nemo
    # rmse_summed = norm(rmse_summed)
    fig, axs = plt.subplots(1,2)

    axs[0].plot(filter_widths, rmse_cls)
    axs[0].plot(filter_widths, rmse_nemo)
    # axs[0].plot(filter_widths, rmse_summed)
    axs[0].grid(b=True, which='major', color='#888888', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs[0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)  
    # axs[0].xlim([0, 900])
    axs[0].minorticks_on()
    # axs[0].ylim([-0.05, 1])
    # axs[0].title('')
    # axs[0].set_xlabel('Gaussian kernel half-width radius (km)')
    # axs[0].set_ylabel('RMSE')
    # axs[0].yscale("log")

    axs[1].plot(filter_widths, ss_cls)
    axs[1].plot(filter_widths, ss_nemo)
    # axs[1].plot(filter_widths, ss_summed)
    axs[1].grid(b=True, which='major', color='#888888', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs[1].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)  
    # axs[1].xlim([0, 900])
    axs[1].minorticks_on()
    # axs[1].ylim([-0.05, 1])
    # axs[1].title('')
    # axs[1].set_xlabel('Gaussian kernel half-width radius (km)')
    # axs[1].set_ylabel('Structural Similarity Index Measurement')
    # axs[1].yscale("log")
    
    plt.show()
    plt.close()

    plt.plot(filter_widths, rmse_cls)
    plt.plot(filter_widths, rmse_nemo)
    # plt.plot(filter_widths, rmse_summed)
    plt.grid(b=True, which='major', color='#888888', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)  
    plt.xlim([100, 700])
    plt.minorticks_on()
    plt.ylim([-0.05, 0.5])
    # plt.title('')
    plt.xlabel('Gaussian kernel half-width radius (km)')
    plt.ylabel('RMSE')
    # plt.yscale("log")
    plt.show()

    plt.plot(filter_widths, ss_cls)
    plt.plot(filter_widths, ss_nemo)
    # plt.plot(filter_widths, ss_summed)
    plt.grid(b=True, which='major', color='#888888', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)  
    plt.xlim([100, 700])
    plt.minorticks_on()
    # plt.ylim([-0.05, 0.5])
    # plt.title('')
    plt.xlabel('Gaussian kernel half-width radius (km)')
    plt.ylabel('Structural Similarity Index Measurement')
    # plt.yscale("log")
    plt.show()


    rmse_pmf = []
    for mdt in pmf_mdts:
        rmse = np.sqrt(np.nanmean(((mdt - cls_mdt)**2)))
        rmse_pmf.append(rmse)
    rmse_pmf = np.array(rmse_pmf)
    rmse_pmf = norm(rmse_pmf)
    plt.plot(iterations, rmse_pmf)
    plt.show()
    

if __name__ == '__main__':
    main()