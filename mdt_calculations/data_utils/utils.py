import numpy as np
from mdt_calculations.data_utils.dat import read_surface


def define_dims(resolution):
    r"""
    Input arguments: resolution
    """
    II = 360 // resolution
    JJ = 180 // resolution
    return int(II), int(JJ)


def create_coords(resolution, central_lon=0, rads=False):
    r"""
    Defines gloibal lon and lat, with lat shifted to
    midpoints and set between -90 and 90.
    """
    II, JJ = define_dims(resolution)
    longitude = np.array([resolution * (i - 0.5) - central_lon for i in range(II)])
    latitude = np.array([resolution * (j - 0.5) - 90.0 for j in range(JJ)])
    if rads:
        longitude = np.deg2rad(longitude)
        latitude = np.deg2rad(latitude)

    return longitude, latitude


def get_res(arr):
    width = max(arr.shape[-1], arr.shape[-2])
    return 360 / width


def bound_arr(arr, lower_bd, upper_bd):
    arr[np.isnan(arr)] = lower_bd
    arr[arr < lower_bd] = lower_bd
    arr[arr > upper_bd] = upper_bd
    return arr


def apply_mask(resolution, surface, mask_filename=None, path=None):
    if path is None:
        path = './fortran/data/src'
    
    if resolution == 0.25:
        mask = read_surface('mask_glbl_qrtd.dat', path)
        surface = surface + mask
        return surface
    elif resolution == 0.5:
        mask = read_surface('mask_glbl_hlfd.dat', path)
        surface = surface + mask
        return surface
    else:
        print("Mask with correct resolution not found.")
        return surface


def calc_residual(arr1, arr2):
    r""" Calculates the residual between two surfaces.

    Checks whether input arrays have the same dimensions.

    Args:
        arr1 (np.array): surface 1.
        arr2 (np.array): surface 2.

    Returns:
        np.array: An array representing the residual surface
            i.e. the difference between the input surfaces.
    """
    if np.shape(arr1) == np.shape(arr2):
        return np.subtract(arr1, arr2)
    else:
        return print("Cannot compute residual: surfaces are not same shape")