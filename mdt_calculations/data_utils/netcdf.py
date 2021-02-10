from netCDF4 import Dataset as netcdf_dataset
import os


def load_cls(fname, path=None):
    if path is None:
        path = ""
    filepath = os.path.join(os.path.normpath(path), fname)
    dataset = netcdf_dataset(filepath)
    mdt = dataset.variables['mdt'][0,:,:]
    lats = dataset.variables['latitude'][:]
    lons = dataset.variables['longitude'][:]
    return mdt, lats, lons


def load_dtu(fname, path=None):
    if path is None:
        path = ""
    filepath = os.path.join(os.path.normpath(path), fname)
    dataset = netcdf_dataset(filepath)
    mdt = dataset.variables['mdt'][:]
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    return mdt, lats, lons
