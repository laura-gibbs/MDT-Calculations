import csv
import numpy as np
import matplotlib.pyplot as plt
import array
#-------------------------------------------------

# define resolution - only need to change this and the file name
#-------------------------------------------------
res = 0.25
#-------------------------------------------------

# determine dimensions
#-------------------------------------------------
II = 360.0/res
JJ = 180.0/res
#-------------------------------------------------

#
#-------------------------------------------------
lns = res
lts = res
#-------------------------------------------------

# read in the data
#-------------------------------------------------
fid=open("./gtim5_do0150_rr0004.dat",mode='r')
a = array.array("i")  
a.fromfile(fid, 1)

b = a[0]
floats = array.array("f") 
floats.fromfile(fid, b//4+1)
floats = floats[1:]
floats = np.asarray(floats)
floats = np.reshape(floats, (II,JJ))
#-------------------------------------------------

# define lon/lat for axes
#-------------------------------------------------
lon = np.zeros(II)
lat = np.zeros(JJ)
for i in range(II):
  lon[i]=round(lns*(i+0.5),2)

for j in range(JJ):
  lat[j]=round(lts*(j-0.5)-90.0,2)
#-------------------------------------------------

# quick plot
#-------------------------------------------------
plt.imshow(floats)
plt.gca().invert_yaxis()
plt.xticks([i for i in range(0,II,II/10)],lon)
plt.yticks([j for j in range(0,JJ,JJ/10)],lat)
plt.show()
#-------------------------------------------------
