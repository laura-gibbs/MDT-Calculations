import numpy as np
import math

torad = math.atan(1.0) / 45.0

degree = np.arange(10.) * 90
print("Degree values : \n", degree)

radian_Values = np.radians(degree)
print("\nRadian values : \n", radian_Values)

radian = np.deg2rad(degree)
print("\nradian values : \n", radian)

to_rads = torad * degree
print("\nradian values : \n", to_rads)
