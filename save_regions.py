
import glob
import os
from mdt_calculations.data_utils.dat import read_surface
from mdt_calculations.data_utils.utils import bound_arr
from mdt_calculations.plotting.plot import plot
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize 


# Make empty array to be appended
x_coords = []
y_coords = []
# For threshold 64 North and South
y0 = (90 - 64) * 4
y1 = (180 - 26) * 4
x0 = 0
x1 = 360 * 4
y = y0
for i in range((y1 - y0) // 128):
    y = i * 128 + y0
    for j in range((x1 - x0) // 128):
        x = j * 128
        x_coords.append(x)
        y_coords.append(y)

mdt = False
if mdt:
    var = 'mdt'
else:
    var = 'cs'

fname = f'../a_mdt_data/computations/currents/dtu18_EIGEN-6C3stat_do0280_rr0004_cs.dat'
    # 'dtu18_XGM2019e_2159_do0280_rr0004.dat',         #2190
    # 'dtu18_SGG-UGM-1_do0280_rr0004.dat',             #2159
    # 'dtu18_EIGEN-6C3stat_do0280_rr0004.dat',         #1949
    # 'dtu18_EIGEN-6C2_do0280_rr0004.dat'              #1949
arr = read_surface(fname)
arr[np.isnan(arr)] = 0
arr = bound_arr(arr, -2, 2)
print(np.min(arr), np.max(arr))
plt.imshow(arr)
plt.show()
arr = resize(arr, (720, 1440), order=2)
# plt.imshow(arr)
# plt.show()
fname = os.path.split(fname)[-1]
fname = fname[:len(fname)-4]
for x, y in zip(x_coords, y_coords):
    region = arr[y:y+128, x:x+128]
    # Naming needs to be fixed
    if mdt:
        nans = np.isnan(arr)
        arr[nans] = 0
    # if (np.counif (np.count_nonzero(region==0)/(region.size)) < 0.25:
    np.save(f'../a_mdt_data/geodetic_cdae/{var}/' + fname + f'_{x}_{y}', region)
    print("valid region", x, y)
    # else:
    #     print("not valid region", x, y)
# filenames = glob.glob('../a_mdt_data/computations/currents/cls18_cs.dat')
# for fname in filenames:
#     arr = read_surface(fname)
#     print(arr.shape)
#     arr = resize(arr, (720, 1440), order=2)
#     print(arr.shape)
#     fname = os.path.split(fname)[-1]
#     fname = fname[:len(fname)-4]
#     for x, y in zip(x_coords, y_coords):
#         region = arr[y:y+128, x:x+128]
#         # Naming needs to be fixed
#         if mdt:
#             nans = np.isnan(arr)
#             arr[nans] = 0
#         # if (np.count_nonzero(region==0)/(region.size)) < 0.25:
#         np.save('../a_mdt_data/HR_model_data/cls18_cs_regions/' + fname + f'_{x}_{y}', region)
#         print("valid region", x, y)
        # else:
        #     print("not valid region", x, y)
        