from mdt_calculations.data_utils.netcdf import load_cls, load_dtu  # load_dtu - pc breaks with big dtu mdts
from mdt_calculations.plotting.plot import plot
from mdt_calculations.data_utils.dat import read_surface, read_surfaces, write_surface
from mdt_calculations.computations.wrapper import mdt_wrapper, cs_wrapper
import numpy as np
import os
import matplotlib.pyplot as plt


def calc_mean(fname, dat_file, path, fig_dir, dat_dir, mean_per='model', plot_bool=False, write=False):
    filepath = os.path.join(os.path.normpath(path), fname)
    f = open(filepath, "r")
    # ignore header
    f.readline()
    params = []
    for line in f.read().splitlines():
        params.append(line.split())
    params = np.array(params)
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


def main():
    mdts = '../a_mdt_data/computations/mdts/'
    cs = '../a_mdt_data/computations/cs/'
    mask = read_surface('mask_rr0004.dat', '../a_mdt_data/computations/masks/')
    hist_cmip5_path = '../a_mdt_data/datasets/cmip5/historical/'
    rcp60 = '../a_mdt_data/datasets/cmip5/rcp60/' 
    hist_cmip5_file = 'cmip5_historical_mdts_yr5_meta.txt'
    hist_cmip5_datfile = 'cmip5_historical_mdts_yr5.dat'

    hist_cmip6_file = 'cmip6_historical_mdts_yr5_meta.txt'
    hist_cmip6_path = '../a_mdt_data/datasets/cmip6/'
    hist_cmip6_datfile = 'cmip6_historical_mdts_yr5.dat'

    means = calc_mean(hist_cmip6_file, hist_cmip6_datfile, hist_cmip6_path, '../a_mdt_data/figs/cmip6/model_means/', '../a_mdt_data/computations/cmip6_calcs/model_means/',
                      mean_per='model')
    

    means = np.array(means)
    means[means > 4] = np.nan
    total_mean = np.nanmean(means, axis=(0))
    total_std = np.nanstd(means, axis=(0))
    total_mean = total_mean.T
    total_std = total_std.T
    # total_std = total_std + mask
    fig = plot(total_mean)#, low_bd=.35, up_bd=0.8)
    fig.set_size_inches((20, 10.25))
    fig.savefig('../a_mdt_data/figs/cmip6/cmip6_historical_mean', dpi=300)
    plt.close()
    write_surface('../a_mdt_data/computations/cmip6_calcs/cmip6_historical_mean', total_mean)
    fig = plot(total_std)#, low_bd=.35, up_bd=0.8)
    fig.set_size_inches((20, 10.25))
    fig.savefig('../a_mdt_data/figs/cmip6/cmip6_historical_std', dpi=300)
    plt.close()
    write_surface('../a_mdt_data/computations/cmip6_calcs/cmip6_historical_std', total_std)

    # stds = np.nanstd(means, axis=(1,2))


    # cmip5_historical = read_surfaces('cmip5_historical_mdts_yr5.dat', historical, number=31, start=0)
    # for i, mdt in enumerate(cmip5_historical):
    #     fig = plot(mdt.T, title=params[i][0]+'_'+params[i][1]+'_'+params[i][2])
    #     fig.set_size_inches((20, 10.25))
    #     fig.savefig('gif_imgs/'+params[i][0]+'_'+params[i][1]+'_'+params[i][2], dpi=300)
    #     plt.close()
        # plt.show()



if __name__ == '__main__':
    main()