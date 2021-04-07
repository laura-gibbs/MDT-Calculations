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
from skimage.util import pad
from scipy.ndimage import convolve


EXTENTS = {
    'na': ((-80, -10), (20, 70)),
    'ag': ((0, 50), (-10, 50)),
}


def boxcar_plot():
    # create empty signal
    x = np.zeros(270)

    # set middle as boxcar window
    x[90:180] = 1

    # create boxcar window
    z = np.ones(90)

    # iteratively apply boxcar window
    y = [x]
    for i in range(3):
        y.append(np.convolve(y[i], z, mode='same') / len(z))

    # plot all signals
    for i in range(len(y)):
        plt.plot(y)
    
    plt.show()

def norm(a):
    return (a - a.min()) / (a.max() - a.min())


def extract_region(mdt, lon_range, lat_range, central_lon=0, central_lat=0):
    res = mdt.shape[0] // 180

    px = ((lon_range[0] + central_lon) * res, (lon_range[1] + central_lon) * res)
    py = ((lat_range[0] + 90) * res, (lat_range[1] + 90) * res)

    return mdt[py[0]:py[1], px[0]:px[1]]


def pmf(arr, iterations=350, k=.16, g=0.02, option=2, suppress=True):
    pmf_arr = []
    its = []
    for i in range(iterations):
        print(f"running iteration {i}")
        filtered_arr = anisotropic_diffusion(arr, niter=1, kappa=k, gamma=g, option=option)
        if suppress:
            test = prevent_gradient_sharpening(arr, filtered_arr)
            # print(np.sum(np.abs(test-arr)), np.sum(np.abs(test-filtered_arr)), np.sum(np.abs(filtered_arr-arr)))
            arr = test
        else:
            arr = filtered_arr
        pmf_arr.append(arr)
        its.append(i)

    return pmf_arr, its


def adjacent_differences(arr):
    # subtract mdt (without edge pixels) from mdt (without edge pixels) shifted in 8 different directions
    return np.array([arr[i%3:arr.shape[0]-2+i%3, i//3:arr.shape[1]-2+i//3] - arr[1:-1, 1:-1] for i in [0, 1, 2, 3, 5, 6, 7, 8]])


def prevent_gradient_sharpening(mdt, filtered_mdt):
    # calculate differences between pixel and its neighbours
    before_diff = adjacent_differences(mdt)
    after_diff = adjacent_differences(filtered_mdt)

    # calculate rmse of differences
    before_rmse = np.sum(before_diff ** 2, axis=0) / 8
    after_rmse = np.sum(after_diff ** 2, axis=0) / 8
    # print(before_rmse.shape, after_rmse.shape)


    out = filtered_mdt.copy()
    # if gradient is sharper then revert pixel change
    out[1:-1, 1:-1] = np.where(
        after_rmse > before_rmse,
        mdt[1:-1, 1:-1],
        filtered_mdt[1:-1, 1:-1]
    )
    # print(np.any(after_rmse > before_rmse), np.sum(after_rmse > before_rmse))
    # print(np.sum(np.abs(out - filtered_mdt)))
    return out


def generate_window(name, size):
    # size is width not radius
    if name == 'boxcar':
        return np.ones((size, size))
    if name == 'hamming':
        return np.outer(np.hamming(size), np.hamming(size))
    if name == 'hanning':
        return np.outer(np.hanning(size), np.hanning(size))


def apply_filter(arr, name, size):
    window = generate_window(name, size)
    # convolve filter window with arr
    arr = convolve(arr, window)
    # divide output by filter sum to enforce weighted average
    arr = arr / np.sum(window)
    return arr



def main():
    mdts = '../a_mdt_data/computations/mdts/'
    gauss_filtered = '../a_mdt_data/computations/mdts/gauss_filtered/'
    cs_path = '../a_mdt_data/computations/currents/'
    gauss_cs_path = '../a_mdt_data/computations/currents/gauss_filtered/'
    masks = '../a_mdt_data/computations/masks/'
    mask = read_surface('mask_rr0008.dat', masks)
    mask_4 = read_surface('mask_rr0004.dat', masks)
    fig_dir = 'figs/mdt_plots/dtu18_gtim5_filtered/'

    dtu18_gtim5_4 = read_surface('sh_mdt_DTU18_GTIM5_L280.dat', mdts)
    dtu18_gtim5 = read_surface('dtu18_gtim5_do0280_rr0008.dat', mdts)
    # dtu18_gtim5 = resize(dtu18_gtim5, (1440, 2880), order=3)
    dtu18_gtim5[np.isnan(dtu18_gtim5)] = 0
    # dtu18_gtim5 = np.clip(dtu18_gtim5, -1.4, 1.4)
    # dtu18_gtim5 = norm(dtu18_gtim5)

    # filt8 = apply_filter(dtu18_gtim5_4, 'boxcar', 8)
    # filt12 = apply_filter(dtu18_gtim5_4, 'boxcar', 12)
    # filt16 = apply_filter(dtu18_gtim5_4, 'boxcar', 16)
    # filt20 = apply_filter(dtu18_gtim5_4, 'boxcar', 20)
    # multi_plot([filt8, filt12, filt16, filt20], extent='na', coastlines=True)


    cls_mdt = read_surface('cls18_mdt.dat', mdts)
    cls_mdt[np.isnan(cls_mdt)] = 0
    cls_downscaled = rescale(cls_mdt, 0.5)
    # cls_downscaled = np.clip(cls_downscaled, -1.4, 1.4)
    # cls_downscaled = norm(cls_downscaled)
    
    nemo_mdt = read_surface('orca0083_mdt.dat', mdts)
    nemo_mdt[np.isnan(nemo_mdt)] = 0
    nemo_downscaled = resize(nemo_mdt, (720, 1440))
    # nemo_downscaled = np.clip(nemo_downscaled, -1.4, 1.4)
    # nemo_downscaled = norm(nemo_downscaled)

    # # Gaussian filter
    filter_widths = []
    gauss_mdts = []
    gauss_mdt_plots = []
    for i in range(32):
        k = (i+1)/4
        mdt = gaussian_filter(dtu18_gtim5, sigma=k)
        mdt[np.isnan(mdt)] = 0
        gauss_mdts.append(mdt)
        r = int((int((4*k)+0.5))/4 * 111)
        filter_widths.append(r)
        if r in (138, 249, 360, 471):
            # resized_mdt = resize(mdt, (1440, 2880), order=3)
            gauss_mdt_plots.append(mdt)
    # multi_plot(gauss_mdt_plots, extent='na', coastlines=True)

    res1 = dtu18_gtim5 - gauss_mdt_plots[0]
    res2 = dtu18_gtim5 - gauss_mdt_plots[1]
    res3 = dtu18_gtim5 - gauss_mdt_plots[2]
    res4 = dtu18_gtim5 - gauss_mdt_plots[3]

    multi_plot([res1, res2, res3, res4], product='resid', extent='na', coastlines=True)
    plt.show()

    # Plot Gaussian filter widths: for r = 5, 9, 13, 17
    # gauss_plots = []
    # for i in range(4):
    #     k = ((i+1)*4) 
        # image = resize(gauss_mdts[k], (1440, 2880), order=3) + mask
        # gauss_plots.append(image)
        # plot(image, product='mdt', extent='na')
        # plt.show()
        # plt.close()

    
    # # Load gaussian filtered currents
    # gauss_cs = []
    # gauss_cs_plots = []
    # for i in range(32):
    #     k = (i+1)/4
    #     r = int((int((4*k)+0.5))/4 * 111)
    #     cs = read_surface(f'dtu18_gtim5_gauss_{r}km_cs.dat', gauss_cs_path)
    #     cs[np.isnan(cs)] = 0
    #     gauss_cs.append(cs)
    #     if r in (138, 249, 360, 471):
    #         resized_cs = resize(cs, (1440, 2880), order=3) + mask
    #         gauss_cs_plots.append(resized_cs)

    
    # multi_plot(gauss_cs_plots, product='cs', extent='na', coastlines=True)
    # multi_plot(gauss_cs_plots, product='cs', extent='na', coastlines=True)

    # pmf_0 = read_surface('dtu18_gtim5_land0_pmf_300i_16k_02g.dat', mdts)
    # pmf_1 = read_surface('dtu18_gtim5_land0_pmf_500i_16k_02g.dat', mdts)
    # pmf_2 = read_surface('dtu18_gtim5_land0_pmf_700i_16k_02g.dat', mdts)
    # pmf_3 = read_surface('dtu18_gtim5_land0_pmf_900i_16k_02g.dat', mdts)

    # pmf_0[np.isnan(pmf_0)] = 0
    # pmf_1[np.isnan(pmf_1)] = 0
    # pmf_2[np.isnan(pmf_2)] = 0
    # pmf_3[np.isnan(pmf_3)] = 0
    # pmf_0 = resize(pmf_0, (1440, 2880), order=3)
    # pmf_1 = resize(pmf_1, (1440, 2880), order=3)
    # pmf_2 = resize(pmf_2, (1440, 2880), order=3)
    # pmf_3 = resize(pmf_2, (1440, 2880), order=3)

    # multi_plot([pmf_0, pmf_1, pmf_2, pmf_3], product='mdt', extent='na', coastlines=True)
    # plt.show()
    

    # PMF Filter
    pmf_mdts, iterations = pmf(dtu18_gtim5, iterations=900)
    extent = 'na'
    # write_surface('dtu18_gtim5_land0_pmf_30i_16k_05g.dat', pmf_mdts[29], mdts)
    # write_surface('dtu18_gtim5_land0_pmf_50i_16k_05g.dat', pmf_mdts[49], mdts)# overwrite=True)
    # write_surface('dtu18_gtim5_land0_pmf_70i_16k_05g.dat', pmf_mdts[69], mdts)# overwrite=True)
    # write_surface('dtu18_gtim5_land0_pmf_90i_16k_05g.dat', pmf_mdts[89], mdts)# overwrite=True)

    res1 = dtu18_gtim5 - pmf_mdts[299]
    res2 = dtu18_gtim5 - pmf_mdts[499]
    res3 = dtu18_gtim5 - pmf_mdts[699]
    res4 = dtu18_gtim5 - pmf_mdts[899]

    multi_plot([res1, res2, res3, res4], product='resid', extent='na', coastlines=True)
    plt.show()

    # get pmf from interior na
    pmf_regions = [extract_region(pmf_mdt, (-60,-40), (10,30)) for pmf_mdt in pmf_mdts] 

    # rmse between pmf and gaussian[25] same extent
    g_region = extract_region(gauss_mdts[25], (-60,-40), (10,30))

    rmses = []
    for pmf_region in pmf_regions:
        rmses.append(np.sqrt(np.nanmean(((g_region - pmf_region)**2))))

    plt.plot(iterations, rmses)
    plt.grid(b=True, which='major', color='#888888', linestyle='-')
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)  
    # plt.xlim([100, 600])
    plt.minorticks_on()
    # plt.ylim([-0.05, 0.5])
    # plt.title('')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE (m)')
    # plt.yscale("log")
    plt.show()

    # Original MDT
    plot(resize(dtu18_gtim5, (1440, 2880), order=3) + mask, product='mdt', extent=extent, coastlines=True)
    plt.show()

    # # Gaussian mdt filtered 200km radius
    # image = resize(gauss_mdts[6], (1440, 2880), order=3) + mask
    # plot(image, product='mdt', extent=extent)
    # plt.show()

    # Gaussian: Residual from original mdt
    gauss_residual = dtu18_gtim5 - gauss_mdts[12]
    image = resize(gauss_residual, (1440, 2880), order=3) + mask
    plot(image, product='mdt', extent=extent, low_bd=-0.15, up_bd=0.15, coastlines=True)
    plt.show()

    # PMF filtered mdt 350 iterations
    image = resize(pmf_mdts[349], (1440, 2880), order=3) + mask
    plot(image, product='mdt', extent=extent, coastlines=True)
    plt.show()

    # PMF: Residual from original mdt
    pmf_residual = dtu18_gtim5 - pmf_mdts[349]
    image = resize(pmf_residual, (1440, 2880), order=3) + mask
    plot(image, product='mdt', extent=extent, low_bd=-0.15, up_bd=0.15, coastlines=True)
    plt.show()
    
    # PMF - gaussian MDT residual
    pmf_gauss = pmf_mdts[349] - gauss_mdts[12]
    image = resize(pmf_gauss, (1440, 2880), order=3) + mask
    plot(image, product='mdt', extent=extent, low_bd=-0.1, up_bd=0.1, coastlines=True)
    plt.show()

    lon_range, lat_range = EXTENTS['na']
    gauss_mdts = [extract_region(mdt, lon_range, lat_range) for mdt in gauss_mdts]
    cls_downscaled = extract_region(cls_downscaled, lon_range, lat_range)
    nemo_downscaled = extract_region(nemo_downscaled, lon_range, lat_range)

    # RMSE
    rmse_cls = []
    ss_cls = []
    for mdt in gauss_mdts:
        # print(mdt.shape, cls_downscaled.shape)
        rmse = np.sqrt(np.nanmean(((mdt - cls_downscaled)**2)))
        rmse_cls.append(rmse)
        ss = ssim(mdt, cls_downscaled)
        ss_cls.append(ss)
    rmse_cls = np.array(rmse_cls)
    rmse_cls = norm(rmse_cls) #+ 0.01
    ss_cls = np.array(ss_cls)

    rmse_nemo = []
    ss_nemo = []
    for mdt in gauss_mdts:
        rmse = np.sqrt(np.nanmean(((mdt - nemo_downscaled)**2)))
        rmse_nemo.append(rmse)
        ss = ssim(mdt, nemo_downscaled)
        ss_nemo.append(ss)
    rmse_nemo = np.array(rmse_nemo)
    rmse_nemo = norm(rmse_nemo) #+ 0.01
    ss_nemo = np.array(ss_nemo)
    
    rmse_summed = rmse_cls + rmse_nemo
    rmse_summed = norm(rmse_summed) + 0.01
    fig, axs = plt.subplots(1,2)

    axs[0].plot(filter_widths, rmse_cls)
    axs[0].plot(filter_widths, rmse_nemo)
    # axs[0].plot(filter_widths, rmse_summed)
    axs[0].grid(b=True, which='major', color='#888888', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs[0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)  
    # axs[0].set_xlim([0, 900])
    axs[0].minorticks_on()
    # axs[0].set_ylim([-0.05, 1])
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
    # axs[1].set_xlim([0, 900])
    axs[1].minorticks_on()
    # axs[1].ylim([-0.05, 1])
    # axs[1].title('')
    axs[1].set_xlabel('Gaussian kernel half-width radius (km)')
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
    # plt.xlim([0, 900])
    plt.minorticks_on()
    # plt.ylim([-0.05, 0.6])
    # plt.title('')
    plt.xlabel('Gaussian filter half-width radius (km)')
    plt.ylabel('RMSE')
    # plt.yscale("log")
    plt.show()

    plt.plot(filter_widths, ss_cls)
    plt.plot(filter_widths, ss_nemo)
    # plt.plot(filter_widths, ss_summed)
    plt.grid(b=True, which='major', color='#888888', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)  
    # plt.xlim([100, 600])
    plt.minorticks_on()
    plt.ylim([-0.05, 0.5])
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