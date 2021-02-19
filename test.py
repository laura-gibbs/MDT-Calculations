from mdt_calculations.data_utils.netcdf import load_cls, load_dtu  # load_dtu - pc breaks with big dtu mdts
from mdt_calculations.plotting.plot import plot
from mdt_calculations.data_utils.dat import read_surface, read_surfaces, write_surface
from mdt_calculations.computations.wrapper import mdt_wrapper, cs_wrapper
import numpy as np
import os
import matplotlib.pyplot as plt


def calc_mean(fname="cmip5_historical_mdts_yr5_meta.txt", path='../a_mdt_data/datasets/cmip5/historical/', fig_dir='figs/cmip/model_means/',
              dat_dir='cmip_calcs/model_means/', mean_per='model'):
    filepath = os.path.join(os.path.normpath(path), fname)
    f = open(filepath, "r")
    f.readline() # ignore header
    params = []
    for line in f.read().splitlines():
        params.append(line.split())

    means = []
    if mean_per == "model":
        models = np.array(params[:,0])
        model_starts = []
        model_names = []
        for i, model in enumerate(models):
            if model not in model_names:
                model_starts.append(i)
                model_names.append(model)
        print(model_starts, model_names)

        for start, end in zip(model_starts[:-1], model_starts[1:]):
            model = models_ens[start][0]
            ensemble = models_ens[start][1]
            batch_size = end - start
            batch = read_surfaces(fname, path, number=batch_size, start=start)
            mean = np.mean(batch, axis=(0))
            means.append(mean)
            fig = plot(mean.T)
            fig.set_size_inches((20, 10.25))
            fig.savefig(fig_dir+model+'_mean', dpi=300)
            write_surface(dat_dir+model+'_mean', mean.T)
            print("mean shape", mean.shape)

    elif mean_per == "ensemble":
        ensemble_starts = [0]
        prev = 0
        for i, year in enumerate(params[:,2]):
            if int(year) < prev:
                ensemble_starts.append(i)
            prev = int(year)

        for i, start in enumerate(ensemble_starts[:-1]):
            batch_size = ensemble_starts[i+1] - ensemble_starts[i]
            model = params[:,0][start]
            ensemble = params[:,1][start]
            batch = read_surfaces(fname, path, number=batch_size, start=start)
            mean = np.mean(batch, axis=(0))
            means.append(mean)
            fig = plot(mean.T)
            fig.set_size_inches((20, 10.25))
            fig.savefig(fig_dir+model+'_'+ensemble+'_mean', dpi=300)
            write_surface(dat_dir+model+'_'+ensemble+'_mean', mean.T)
            print("mean shape", mean.shape)


def main():
    mdts = '../a_mdt_data/computations/mdts/'
    cs = '../a_mdt_data/computations/cs/'
    historical = '../a_mdt_data/datasets/cmip5/historical/'
    rcp60 = '../a_mdt_data/datasets/cmip5/rcp60/' 

    # cls18_mdt = read_surface('cls18_mdt.dat', mdts)
    # print("cls shape", cls18_mdt.shape)
    # # plot(cls18_mdt, title='CLS18 MDT 1/8th degree')
    # nemo_mdt = read_surface('orca0083_mdt_12th.dat', mdts)
    # # plot(nemo_mdt, title='Nemo MDT, 1/12th degree')

    # cls18_cs = read_surface('cls18_cs.dat', cs)
    # # plot(cls18_cs, title='CLS18 Geostrophic Currents', product='cs')

    # nemo_cs = read_surface('orca0083_cs.dat', cs)
    # plot(nemo_cs, bds=1.4, title='Nemo Geostrophic Currents', product='cs')

    # path2 = './../a_mdt_data/computations/currents/'
    # cs = read_surface('shmdtout.dat', path2)
    # # plot(cs, bds=1.5, product='cs')

    # hadgem2_r1_2001 = read_surfaces('cmip5_historical_mdts_yr5.dat', historical, number=1, start=3784) 
    # plot(hadgem2_r1_2001[0].T, title='HADGEM2-ES r1i1p1 MDT (2001-2006), 1/4 degree')
 
    # dtu = read_surface('dtu10mdt_1min.dat', path0)
    # plot(dtu_mdt, lats=lats, lons=lons)

    # access1_0 = read_surfaces('cmip5_historical_mdts_yr5.dat', historical, number=30, start=0)
    # can_CM4 = read_surfaces('cmip5_historical_mdts_yr5.dat', historical, number=9, start=341)

    file = "cmip5_historical_mdts_yr5_meta.txt"
    filepath = os.path.join(os.path.normpath(historical), file)
    f = open(filepath, "r")
    f.readline() # ignore header
    params = []
    for line in f.read().splitlines():
        params.append(line.split())

    params = np.array(params)
    models_ens = params[:, (0,1)]
    models = params[:, 0]
    print(models)
    model_starts = []
    model_names = []
    for i, model in enumerate(models):
        if model not in model_names:
            model_starts.append(i)
            model_names.append(model)
    print(model_starts, model_names)

    start_arr = [0]
    prev = 0
    for i, year in enumerate(params[:,2]):
        if int(year) < prev:
            start_arr.append(i)
        prev = int(year)

    # mask = read_surface('mask_rr0004.dat', '../a_mdt_data/computations/masks/')
    # means = []
    # for start, end in zip(model_starts[:-1], model_starts[1:]):
    #     model = models_ens[start][0]
    #     ensemble = models_ens[start][1]
    #     print(model, ensemble)
    #     batch_size = end - start
    #     batch = read_surfaces('cmip5_historical_mdts_yr5.dat', historical, number=batch_size, start=start)
    #     mean = np.mean(batch, axis=(0))
    #     # mean = mean + mask
    #     means.append(mean)
    #     # write_surface('cmip_calcs/model_means_mask/'+model+'_mean', mean)
    #     # fig = plot(mean)
    #     # fig.set_size_inches((20, 10.25))
    #     # fig.savefig('figs/cmip/model_means_mask/'+model+'_mean', dpi=300)
    #     # print("means len", len(means))

    # means = np.array(means)
    # means[means > 4] = np.nan
    # # print(means.shape)
    # # total_mean = np.nanmean(means, axis=(0))
    # total_std = np.nanstd(means, axis=(0))
    # print(total_std.shape)
    # # print(total_mean.shape)
    # # mask = read_surface('mask_rr0004.dat', '../a_mdt_data/computations/masks/')
    # # total_mean = total_mean.T
    # total_std = total_std.T
    # total_std = total_std + mask
    # fig = plot(total_std, low_bd=.35, up_bd=0.8)
    # fig.set_size_inches((20, 10.25))
    # # fig.savefig('figs/cmip/cmip5_historical_mean', dpi=300)
    # # write_surface('cmip_calcs/cmip5_historical_mean', total_mean)


    # stds = np.nanstd(means, axis=(1,2))
    # print(list(zip(model_names, stds)))

    # # plt.boxplot(means.reshape(means.shape[0], means.shape[1] * means.shape[2]), showfliers=False)
    # # Find MIROC5 index and remove element from model names and stds
    # outlier = model_names.index('MIROC5')
    # del(model_names[outlier])
    # stds = np.delete(stds, outlier)
    # plt.plot(list(range(42)), stds)
    # # plt.errorbar()
    # plt.show()

    # mask = read_surface('mask_rr0004.dat', '../a_mdt_data/computations/masks/')
    # tbf = read_surface('MIROC5_mean.dat', './cmip_calcs/model_means/')
    # print(np.nanmin(tbf), np.nanmax(tbf))
    # tbf = tbf + mask
    # print(np.nanmin(tbf), np.nanmax(tbf))
    # fig = plot(tbf, low_bd=0, up_bd=25)
    # fig.set_size_inches((20, 10.25))
    print(start_arr)
    return
    print(params)
    cmip5_historical = read_surfaces('cmip5_historical_mdts_yr5.dat', historical, number=31, start=0)
    print(cmip5_historical.shape)
    for i, mdt in enumerate(cmip5_historical):
        fig = plot(mdt.T, title=params[i][0]+'_'+params[i][1]+'_'+params[i][2])
        fig.set_size_inches((20, 10.25))
        fig.savefig('gif_imgs/'+params[i][0]+'_'+params[i][1]+'_'+params[i][2], dpi=300)
        plt.close()
        # plt.show()




if __name__ == '__main__':
    main()