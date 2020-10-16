import numpy as np


def define_dims(res):
    r"""
    res is resolution
    """
    II = 360 // res
    JJ = 180 // res
    return int(II), int(JJ)


def create_coords(res, rads=False):
    r"""
    Defines gloibal lon and lat, with lat shifted to
    midpoints and set between -90 and 90.
    """
    II, JJ = define_dims(res)
    longitude = np.array([res * (i - 0.5) for i in range(II)])
    latitude = np.array([res * (j - 0.5) - 90.0 for j in range(JJ)])
    if rads:
        longitude = np.deg2rad(longitude)
        latitude = np.deg2rad(latitude)

    return longitude, latitude


def bound_arr(arr, lower_bd, upper_bd):
    arr[np.isnan(arr)] = lower_bd
    arr[arr < lower_bd] = lower_bd
    arr[arr > upper_bd] = upper_bd
    return arr
