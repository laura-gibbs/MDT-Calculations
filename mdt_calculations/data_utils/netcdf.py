from netCDF4 import Dataset as netcdf_dataset
import os


def load_cls(fname, path=None, var='mdt'):
    if path is None:
        path = ""
    filepath = os.path.join(os.path.normpath(path), fname)
    dataset = netcdf_dataset(filepath)
    var = dataset.variables[var][0,:,:]
    lats = dataset.variables['latitude'][:]
    lons = dataset.variables['longitude'][:]
    return var, lats, lons


def load_dtu(fname, path=None, var='mdt'):
    if path is None:
        path = ""
    filepath = os.path.join(os.path.normpath(path), fname)
    dataset = netcdf_dataset(filepath)
    print(dataset.variables)
    var = dataset.variables[var][:]
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    return var, lats, lons
