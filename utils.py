import numpy as np


def define_dims(resolution):
    r"""
    Input arguments: resolution
    """
    II = 360 // resolution
    JJ = 180 // resolution
    return int(II), int(JJ)


def create_coords(resolution, rads=False):
    r"""
    Defines gloibal lon and lat, with lat shifted to
    midpoints and set between -90 and 90.
    """
    II, JJ = define_dims(resolution)
    longitude = np.array([resolution * (i - 0.5) for i in range(II)])
    latitude = np.array([resolution * (j - 0.5) - 90.0 for j in range(JJ)])
    if rads:
        longitude = np.deg2rad(longitude)
        latitude = np.deg2rad(latitude)

    return longitude, latitude


def bound_arr(arr, lower_bd, upper_bd):
    arr[np.isnan(arr)] = lower_bd
    arr[arr < lower_bd] = lower_bd
    arr[arr > upper_bd] = upper_bd
    return arr
