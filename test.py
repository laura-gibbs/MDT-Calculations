from mdt_calculations.data_utils.netcdf import load_cls, load_dtu  # load_dtu - pc breaks with big dtu mdts
from mdt_calculations.plotting.plot import plot
from mdt_calculations.data_utils.dat import read_surface, read_surfaces, write_surface
from mdt_calculations.computations.wrapper import mdt_wrapper, cs_wrapper
import numpy as np

def main():
    mdts = '../a_mdt_data/computations/mdts/'
    cs = '../a_mdt_data/computations/cs/'
    historical = '../a_mdt_data/datasets/cmip5/historical/'
    rcp60 = '../a_mdt_data/datasets/cmip5/rcp60/' 


    # rcp60s = read_surfaces('cmip5_rcp60_mdts_yr5.dat', rcp60, number=2, start=0)
    # bcc_r1_2006 = rcp60s[0].T
    # plot(bcc_r1_2006, bds=1.5)
    # bcc_cs = read_surface('bss_r1_2006_cs.dat', path1)
    # plot(bcc_cs, bds=1.0, central_lon=180, product='cs', extent='gs')

    cls18_mdt = read_surface('cls18_mdt.dat', mdts)
    # plot(cls18_mdt, title='CLS18 MDT 1/8th degree')

    nemo_mdt = read_surface('orca0083_mdt_12th.dat', mdts)
    # plot(nemo_mdt, title='Nemo MDT, 1/12th degree')

    hadgem2_r1_2001 = read_surfaces('cmip5_historical_mdts_yr5.dat', historical, number=1, start=3784) 
    plot(hadgem2_r1_2001[0].T, title='HADGEM2-ES r1i1p1 MDT (2001-2006), 1/4 degree')

    cls18_cs = read_surface('cls18_cs.dat', cs)
    # plot(cls18_cs, title='CLS18 Geostrophic Currents', product='cs')

    nemo_cs = read_surface('orca0083_cs.dat', cs)
    plot(nemo_cs, bds=1.4, title='Nemo Geostrophic Currents', product='cs')

    path2 = './../a_mdt_data/computations/currents/'
    cs = read_surface('shmdtout.dat', path2)
    # plot(cs, bds=1.5, product='cs')


if __name__ == '__main__':
    main()