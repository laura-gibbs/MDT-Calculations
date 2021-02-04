from netCDF4 import Dataset as netcdf_dataset


def load_cls(fname):
    dataset = netcdf_dataset(fname)
    mdt = dataset.variables['mdt'][0,:,:]
    lats = dataset.variables['latitude'][:]
    lons = dataset.variables['longitude'][:]
    return mdt, lats, lons


def load_dtu(fname):
    dataset = netcdf_dataset(fname)
    mdt = dataset.variables['mdt'][:]
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    return mdt, lats, lons
