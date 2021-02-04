import math
import numpy as np
import os
import array
import matplotlib.pyplot as plt
from utils import define_dims, create_coords, bound_arr
from read_data import read_surface, read_surfaces, write_surface, apply_mask
import turbo_colormap_mpl

path1 = '../data/src/'
mask_r8 = read_surface('mask_rr0008.dat', path1)
print(mask_r8.dtype)
mask_r8 = 1 - mask_r8   # OR 1 - mask_r8
mask_r8[mask_r8 == 1] = np.nan
print(mask_r8.dtype)
write_surface('fixedmask_rr0008.dat', mask_r8, path=path1, fortran=True, overwrite=True)