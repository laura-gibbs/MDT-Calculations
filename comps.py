from mdt_calculations.data_utils.netcdf import load_cls, load_dtu  # load_dtu - pc breaks with big dtu mdts
from mdt_calculations.plotting.plot import plot, save_figs
from mdt_calculations.data_utils.dat import read_surface, read_surfaces, write_surface, read_params
from mdt_calculations.computations.wrapper import mdt_wrapper, cs_wrapper
from mdt_calculations.plotting.gifmaker import gen_gif
import numpy as np
import os
import matplotlib.pyplot as plt


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

def main():
    mdts = '../a_mdt_data/computations/mdts/'
    cs = '../a_mdt_data/computations/currents/'
    figs_dir = '../a_mdt_data/figs/'
    mask = read_surface('mask_rr0004.dat', '../a_mdt_data/computations/masks/')
    cmip5_path = '../a_mdt_data/datasets/cmip5/historical/'
    rcp60 = '../a_mdt_data/datasets/cmip5/rcp60/' 
    cmip5_file = 'cmip5_historical_mdts_yr5_meta.txt'
    cmip5_datfile = 'cmip5_historical_mdts_yr5.dat'

    cmip6_file = 'cmip6_historical_mdts_yr5_meta.txt'
    cmip6_path = '../a_mdt_data/datasets/cmip6/'
    cmip6_datfile = 'cmip6_historical_mdts_yr5.dat'
    cmip5_models = '../a_mdt_data/computations/cmip5_calcs/model_means/'
    # cmip6_hist = read_surfaces('cmip6_historical_mdts_yr5.dat', cmip6_path, number=32, start=32)
    # mean_mdt = np.nanmean(cmip6_hist, axis=(0))
    # fig = plot(mean_mdt, bds=3)
    # fig.set_size_inches((20, 10.25))

    # means = calc_mean(cmip6_file, cmip6_datfile, cmip6_path, '../a_mdt_data/figs/cmip6/model_means/', '../a_mdt_data/computations/cmip6_calcs/model_means/',
    #                   mean_per='model')
    # total_mean, total_std = compute_std(means, '../a_mdt_data/figs/cmip6/', '../a_mdt_data/computations/cmip6_calcs/', 'cmip6_hist')

    # params = read_params(cmip6_file, cmip6_path)
    # save_figs(cmip6_datfile, cmip6_path, 32, 5120, params, 'gifs/gif_imgs/cmip6MIP-ESM1-2-HR_180/')

    # gen_gif('cmip6MIP-ESM1-2-HR_180', 'cmip6MIP-ESM1-2-HR_180')

    cls18_cs = read_surface('cls18_cs.dat', cs)
    fig = plot(cls18_cs, bds=2, product='cs')
    # fig.savefig(figs_dir+'cls/cls18_cs', dpi=300)

    orca_cs = read_surface('orca0083_cs.dat', cs)
    fig = plot(orca_cs, bds=2, product='cs')
    # fig.savefig(figs_dir+'nemo/nemo_cs', dpi=300)

    access_cs = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
    fig = plot(access_cs, bds=2, product='cs', log=True)
    fig.savefig(figs_dir+'cmip6/cs/MPI-ESM1-2-HR_cs_log', dpi=300)
    plt.close()
    access_cs = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
    fig = plot(access_cs, bds=2, product='cs', extent='gs', log=True)
    fig.savefig(figs_dir+'cmip6/cs/MPI-ESM1-2-HR_gs_log', dpi=300)
    plt.close()
    access_cs = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
    fig = plot(access_cs, bds=2, product='cs', extent='ag', log=True)
    fig.savefig(figs_dir+'cmip6/cs/MPI-ESM1-2-HR_ag_log', dpi=300)
    plt.close()
    access_cs = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
    fig = plot(access_cs, bds=2, product='cs')
    fig.savefig(figs_dir+'cmip6/cs/MPI-ESM1-2-HR_cs', dpi=300)
    plt.close()
    access_cs = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
    fig = plot(access_cs, bds=2, product='cs', extent='gs')
    fig.savefig(figs_dir+'cmip6/cs/MPI-ESM1-2-HR_gs', dpi=300)
    plt.close()
    access_cs = read_surface('MPI-ESM1-2-HR_cs.dat', cs)
    fig = plot(access_cs, bds=2, product='cs', extent='ag')
    fig.savefig(figs_dir+'cmip6/cs/MPI-ESM1-2-HR_ag', dpi=300)
    plt.close()
    

    

if __name__ == '__main__':
    main()