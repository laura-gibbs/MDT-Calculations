from mdt_calculations.data_utils.netcdf import load_cls, load_dtu  # load_dtu - pc breaks with big dtu mdts
from mdt_calculations.plotting.plot import plot, save_figs
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

    # params = read_params(cmip6_file, cmip6_path)
    # save_figs(cmip6_datfile, cmip6_path, 32, 5120, params, 'gifs/gif_imgs/cmip6MIP-ESM1-2-HR_180/')

    # gen_gif('cmip6MIP-ESM1-2-HR_180', 'cmip6MIP-ESM1-2-HR_180')

    cls18_cs = read_surface('cls18_cs.dat', cs)
    # fig = plot(cls18_cs, bds=2, product='cs')
    # # fig.savefig(figs_dir+'cls/cls18_cs', dpi=300)

    orca_cs = read_surface('orca0083_cs.dat', cs)
    # fig = plot(orca_cs, bds=2, product='cs', )
    # # fig.savefig(figs_dir+'nemo/nemo_cs', dpi=300)

    access_cs = read_surface('ACCESS1-0_cs.dat', cs)
    mpi_esm1_cs = read_surface('MPI-ESM1-2-HR_cs.dat', cs)

    surface1 = np.array(cls18_cs)
    surface2 = np.array(orca_cs)
    surface3 = np.array(mpi_esm1_cs)
    surface4 = np.array(access_cs)
    # print(surface1.shape, surface2.shape)
    surfaces = np.asarray((surface1, surface2, surface3, surface4))
    # multi_plot(surfaces, 1, 2, extent='gs')
    axtitles = ['CNES-CLS18', 'Nemo ORCA12', 'MPI-ESM1-2-HR (CMIP6)', 'ACCESS1-0 (CMIP5)']
    # multi_plot(surfaces, axtitles=axtitles)

    # arr1 = np.array(read_surface('dtu15_eigen-6c4_do0050_rr0004.dat', mdts))
    # arr2 = np.array(read_surface('dtu15_eigen-6c4_do0100_rr0004.dat', mdts))
    # arr3 = np.array(read_surface('dtu15_eigen-6c4_do0150_rr0004.dat', mdts))
    # arr4 = np.array(read_surface('dtu15_eigen-6c4_do0280_rr0004.dat', mdts))
    # arrs = np.asarray((arr1, arr2, arr3, arr4))
    # titles = ['Degree/order 50', 'Degree/order 100', 'Degree/order 150', 'Degree/order 280']
    # # multi_plot(arrs, product='geoid', axtitles=titles, coastlines=True)
    
    # a_titles = ['DTU15MSS-GTIM5 MDT (degree/order 280)', 'DTU15MSS-EIGEN6C4 MDT (degree/order 280)']
    # arr5 = np.array(read_surface('dtu15_gtim5_do0280_rr0004.dat', mdts))
    # arrs2 = np.asarray((arr5, arr4))
    # # multi_plot(arrs2, product='mdt', axtitles=a_titles, stacked=False)

    # arr6 = np.array(read_surface('dtu15_gtim5_do0280_rr0004_cs.dat', cs))
    # arr7 = np.array(read_surface('dtu15_eigen-6c4_do0280_rr0004_cs.dat', cs))
    # b_titles = ['Geostrophic Currents Produced from the DTU15MSS-GTIM5 MDT (degree/order 280)', 'Geostrophic Currents Produced from the DTU15MSS-EIGEN6C4 MDT (degree/order 280)']
    # arrs3 = np.asarray((arr6, arr7))
    # # multi_plot(arrs3, product='cs', axtitles=b_titles, stacked=False)

    
    # dtu15_eigen6c4 = read_surface('dtu15_ITSG-Grace2018s_do0190_rr0004.dat', mdts)
    # print(np.nanmin(dtu15_eigen6c4), np.nanmax(dtu15_eigen6c4))
    # fig = plot(dtu15_eigen6c4)
    # fig.savefig(figs_dir+'/mdt_plots/geodetic/dtu15_ITSG-Grace2018s_do0190_rr0004', dpi=300)    
    # plt.close()   

    # dtu15_0190 = read_surface('dtu15_do0190_rr0004.dat', mss)
    # fig = plot(dtu15_0190, cmap='inferno', bds=85)
    # fig.savefig(figs_dir+'mss_plots/dtu15_do0190_rr0004', dpi=300)    
    # plt.close()

    # eigen6c4 = read_surface('ITSG-Grace2018s_do0190_rr0004.dat', geoids)
    # fig = plot(eigen6c4, cmap='inferno', bds=85)
    # fig.savefig(figs_dir+'geoid_plots/ITSG-Grace2018s_do0190_rr0004', dpi=300)
    # plt.close()

    # dtu15_eigen6c4_cs = read_surface('dtu15_eigen-6c4_do0280_rr0004_cs.dat', cs)
    # fig = plot(dtu15_eigen6c4_cs, bds=2, product='cs')
    # fig.savefig(figs_dir+'currents/geodetic/dtu15_eigen-6c4_do0280_rr0004_cs', dpi=300)
    # plt.close()

    mask60 = read_surface('gebco/bin/gebco14_1min_lmsk.dat', masks)
    tmp = mask60[:,:10800].copy()
    tmp2 = mask60[:,10800:].copy()
    mask60 = np.concatenate([tmp2, tmp], axis=1)

    dtu_path = '../a_mdt_data/datasets/dtu/'
    # dtu_mss, lats, lons = load_dtu(, dtu_path)
    # print(dtu_mss.shape, lats.shape, lons.shape)
    # write_surface('DTU18MSS_1min.dat', dtu_mss, mss)
    # dtu18MSS = read_surface('dtu18mss_1min.dat', mss)
    # fig = plot(dtu18MSS)
    # plt.show()
    filepath = os.path.join(os.path.normpath(dtu_path), 'DTU18MSS_1min.nc')
    dataset = netcdf_dataset(filepath)
    print(dataset)
    dtu18mss = dataset.variables['mss'][:]
    print(dtu18mss.shape)
    dtu18mss = dtu18mss[:,1:21601] 
    print(dtu18mss.shape)
    # plot(dtu18mss, product='geoid')
    # plt.show()
    # write_surface('dtu18_mss.dat', dtu18mss, mss, overwrite=True)
    # mss = mdt*mask60
    print(dtu18mss.shape)
    # plot(dtu18mss)
    dtu15_cs = read_surface('dtu15_mdt_cs.dat', cs)
    plot(dtu15_cs, product='cs')
    plt.show()

if __name__ == '__main__':
    main()