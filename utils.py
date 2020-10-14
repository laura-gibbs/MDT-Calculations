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
    glon = np.array([res * (i - 0.5) for i in range(II)])
    glat = np.array([res * (j - 0.5) - 90.0 for j in range(JJ)])
    if rads:
        glon = np.deg2rad(glon)
        glat = np.deg2rad(glat)

    return glon, glat
