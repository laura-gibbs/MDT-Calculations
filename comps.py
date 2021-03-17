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
    # cmip6_hist = read_surfaces('cmip6_historical_mdts_yr5.dat', cmip6_path, number=32, start=32)
    # mean_mdt = np.nanmean(cmip6_hist, axis=(0))
    # fig = plot(mean_mdt, bds=3)
    # fig.set_size_inches((20, 10.25))

    # means = calc_mean(cmip6_file, cmip6_datfile, cmip6_path, '../a_mdt_data/figs/cmip6/model_means/', '../a_mdt_data/computations/cmip6_calcs/model_means/',
    #                   mean_per='model')
    # total_mean, total_std = compute_std(means, '../a_mdt_data/figs/cmip6/', '../a_mdt_data/computations/cmip6_calcs/', 'cmip6_hist')

    # # fig.savefig(figs_dir+'cls/cls18_cs', dpi=300)
    
    dtu18_mdt = read_surface('dtu18_eigen-6c4_do0280_rr0004.dat', mdts)
    # plot(dtu18_mdt)
    # plt.show()
    dtu18_cs = read_surface('dtu18_eigen-6c4_do0280_rr0004_cs.dat', cs)
    # plot(dtu18_cs, product='cs')
    # plt.show()
    # plt.close()
    # plot_projection(dtu18_cs, crs=ccrs.Geostationary())


    arr1 = np.array(read_surface('dtu18_gtim5_do0100_rr0004.dat', mdts))
    arr2 = np.array(read_surface('dtu18_gtim5_do0150_rr0004.dat', mdts))
    arr3 = np.array(read_surface('dtu18_gtim5_do0200_rr0004.dat', mdts))
    arr4 = np.array(read_surface('dtu18_gtim5_do0280_rr0004.dat', mdts))
    arrs = np.asarray((arr1, arr2, arr3, arr4))
    titles = ['Degree/order 100', 'Degree/order 150', 'Degree/order 200', 'Degree/order 280']
    multi_plot(arrs, product='mdt', axtitles=titles)

    arr5 = np.array(read_surface('dtu18_gtim5_do0280_rr0004.dat', mdts))
    arr6 = np.array(read_surface('dtu18_eigen-6c4_do0280_rr0004.dat', mdts))
    arrs2 = np.asarray((arr5, arr6))
    titles2 = ['DTU18MSS-GTIM5 MDT (degree/order 280)', 'DTU18MSS-EIGEN6C4 MDT (degree/order 280)']
    multi_plot(arrs2, product='mdt', axtitles=titles2)#, coastlines=True)

if __name__ == '__main__':
    main()