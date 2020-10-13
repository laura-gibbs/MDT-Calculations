import math
import numpy as np
import os
import array
import matplotlib.pyplot as plt

path0 = '.\\fortran\\data\\src\\'
path1 = '.\\fortran\\data\\res\\'

# Reads params from parameters.txt below
# open(21,file='./parameters.txt',form='formatted')
# read(21,'(A20)')name_mss
# read(21,'(I4)')NN_mss
# read(21,'(A20)')name_egm
# read(21,'(I4)')NN_egm
# read(21,'(A4)')tide_in
# read(21,'(I4)')rr
# read(21,'(I4)')NN
# close(21)
# !--------------------
rr = 4 # assumption

#Required output grid
#Steps
lon_stp = 1.0/rr                # Longitude interval
ltgd_stp = 1.0/rr                # Latitude interval

lon_min = 0.5 * lon_stp            # Min longitude
lon_max = 360.0 - 0.5 * lon_stp      # Max longitude

ltgd_min = -90.0 + 0.5 * ltgd_stp   # Min latitude
ltgd_max = 90.0 - 0.5 * ltgd_stp   # Max latitude

# Compute grid dimensions
# and make memory allocations 
II = round((lon_max - lon_min) / lon_stp) + 1       # 1440
JJ = round((ltgd_max - ltgd_min) / ltgd_stp) + 1    # 720


# Calculate longitude and geodetic latitude 
# arrays for points on output grid
lon = np.zeros(II)
lat = np.zeros(JJ)

for i in range(II):
    lon[i] = lon_stp * i + lon_min

for j in range(JJ):
    lat[j] = ltgd_stp * j + ltgd_min






# def main():

# if __name__ == '__main__':
#     main()
