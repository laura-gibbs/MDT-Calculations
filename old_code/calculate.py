import numpy as np
import array
import os

tide_out = 'mean'

# Reference ellipsoid paramters

# GRS80 (from ICGEM)
a = 6378137.0            # Equatorial raidus (m)
f = 1.0/298.257202101  # Recipricol of flattening
gm = 3.986005e14           # Gravity mass constant
omega = 7.292115e-5        # Rotation rate


# integer  :: i,j,n,m,II,JJ
# integer  :: NN_mss,NN_egm,NN,rr


pin = "..\\data\\src\\"
pout = "..\\data\\res\\"


f = open("parameters.txt", 'r')
params = f.read().splitlines()

name_mss = params[0]
NN_mss = int(params[1])
name_egm = params[2]
NN_egm = int(params[3])
tide_in = params[4]
rr = int(params[5])
NN = int(params[6])

f.close()


# Required output grid

lon_stp = 1.0/rr                # Longitude interval
ltgd_stp = 1.0/rr                # Latitude interval
lon_min = 0.5*lon_stp            # Min longitude
lon_max = 360.0-0.5*lon_stp      # Max longitude
ltgd_min = -90.00+0.5*ltgd_stp   # Min latitude
ltgd_max = 90.00-0.5*ltgd_stp   # Max latitude


# Compute grid dimensions and make memory allocations

II = int((lon_max-lon_min)/lon_stp)+1
JJ = int((ltgd_max-ltgd_min)/ltgd_stp)+1

gh = np.zeros((II))
geoid = np.zeros((II, JJ))
lon = np.zeros((II))
lat = np.zeros((JJ))
c_n = np.zeros((NN_egm, NN_egm))
s_n = np.zeros((NN_egm, NN_egm))
c_ref = np.zeros((NN_egm, NN_egm))

for i in range(II):
    lon[i] = lon_stp * (i - 1) + lon_min
 
for j in range(JJ):
    lat[j] = ltgd_stp * (j - 1) + ltgd_min


fname = (os.path.join(pin, name_egm+".dat"))
f = open(fname, 'rb')
# a = array.array("f")
# a.fromfile(f, 2)
# print(a[0], a[1])

floats = np.frombuffer(f.read(), dtype=np.double)
shc = np.reshape(floats, (NN_egm, NN_egm, 4))
print(shc.shape)

# c_n = shc((NN_egm, NN_egm, 1)) 
# s_n = shc((NN_egm, NN_egm, 2))
# c_n(0,0) = 