from mdt_calculations.data_utils.netcdf import load_cls  # load_dtu - pc breaks with big dtu mdts
from mdt_calculations.plotting.plot import plot
from mdt_calculations.data_utils.dat import read_surface, read_surfaces, write_surface
from mdt_calculations.computations.wrapper import mdt_wrapper, cs_wrapper

def main():
    path = 'fortran/data/res/'
    historical = '../cmip5/historical/'
    rcp60 = '../cmip5/rcp60/'

    # historical_cmip5 = read_surfaces('cmip5_historical_mdts_yr5.dat',
    #                                  historical, number=2, start=3786)
    # hadgem_r1_2001 = historical_cmip5[1].T
    # print(hadgem_r1_2001.shape, hadgem_r1_2001.dtype)
    # plot(hadgem_r1_2001)
    # hadGEM2_r1i1p1_2001 = read_surface('')

    # rcp60s = read_surfaces('cmip5_rcp60_mdts_yr5.dat', rcp60, number=2, start=0)
    # bcc_r1_2006 = rcp60s[0].T
    # plot(bcc_r1_2006, bds=1.5)

    # fname = ('../Aviso/mdt/MDT_CNES_CLS18_global.nc')
    # mdt, lons, lats = load_cls(fname)
    # cls_mdt = read_surface('cls18_mdt.dat', path)
    # plot(cls_mdt, bds=1.4)  #, title='CLS18 MDT')

    # path1 = 'fortran/data/'
    # cs = read_surface('cls18_cs.dat', path1)
    # plot(cs, bds=2.0, cmap='jet', central_lon=180, product='cs')#, extent='gs')

    # bcc_cs = read_surface('bss_r1_2006_cs.dat', path1)
    # plot(bcc_cs, bds=1.0, central_lon=180, product='cs', extent='gs')

    path1 = './../a_mdt_data/computations/mdts/'
    mdt = read_surface('dtu15_gtim5_do0280_rr0004.dat', path1)
    plot(mdt)

    path2 = './../a_mdt_data/computations/currents/'
    cs = read_surface('shmdtout.dat', path2)
    plot(cs, bds=1.0, central_lon=180, product='cs')


if __name__ == '__main__':
    main()