from mdt_calculations.data_utils.netcdf import load_cls, load_dtu  # load_dtu - pc breaks with big dtu mdts
from mdt_calculations.plotting.plot import plot
from mdt_calculations.data_utils.dat import read_surface, read_surfaces, write_surface
from mdt_calculations.computations.wrapper import mdt_wrapper, cs_wrapper
import numpy as np
import os


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

    file = "cmip5_historical_mdts_yr5_meta.txt"
    filepath = os.path.join(os.path.normpath(historical), file)
    f = open(filepath, "r")
    f.readline() # ignore header
    params = []
    for line in f.read().splitlines():
        params.append(line.split())

    params = np.array(params)
    models_ens = params[:,(0,1)]
    models = params[:,0]
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


    means = []
    for start, end in zip(model_starts[:-1], model_starts[1:]):
        model = models_ens[start][0]
        ensemble = models_ens[start][1]
        print(model, ensemble)
        batch_size = end - start
        batch = read_surfaces('cmip5_historical_mdts_yr5.dat', historical, number=batch_size, start=start)
        mean = np.mean(batch, axis=(0))
        means.append(mean)
        fig = plot(mean.T)
        fig.set_size_inches((20, 10.25))
        fig.savefig('figs/cmip/model_means/'+model+'_mean', dpi=300)
        write_surface('cmip_calcs/model_means/'+model+'_mean', mean.T)
        print("mean shape", mean.shape)

    # access1_0 = read_surfaces('cmip5_historical_mdts_yr5.dat', historical, number=30, start=0)
    # can_CM4 = read_surfaces('cmip5_historical_mdts_yr5.dat', historical, number=9, start=341)

    # print(access1_0.shape)
    # mean = np.mean(access1_0, axis=(0))
    # print(mean.shape)
    # fig = plot(mean.T)
    # fig.savefig('bla')

    #git test
    


if __name__ == '__main__':
    main()