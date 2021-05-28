
import glob
import os
from mdt_calculations.data_utils.dat import read_surface
from mdt_calculations.plotting.plot import plot
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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

filenames = glob.glob('../a_mdt_data/HR_model_data/nemo_currents/*.dat')
for fname in filenames:
    arr = read_surface(fname)
    fname = os.path.split(fname)[-1]
    fname = fname[:len(fname)-4]
    for x, y in zip(x_coords, y_coords):
        region = arr[y:y+128, x:x+128]
        # Naming needs to be fixed
        if (np.count_nonzero(region==0)/(region.size)) < 0.0125:
            np.save('../a_mdt_data/HR_model_data/qtrland_nemo/' + fname + f'_{x}_{y}', region)
            print("valid region", x, y)
        else:
            print("not valid region", x, y)
            # img = (region - np.nanmin(region)) / (np.nanmax(region) - np.nanmin(region))
            # img = (img * 255).astype('uint8')
            # img = Image.fromarray(img)
            # img.save(fname + f'_{x}' + f'_{y}.png')
            # pass
