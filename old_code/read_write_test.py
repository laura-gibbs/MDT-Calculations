import numpy as np
import matplotlib.pyplot as plt
import read_data as rd


# def read_dat(path, filename, shape, transpose=False):
#     filepath = (os.path.join(path, filename))
#     fid = open(filepath, mode='rb')
#     a = array.array("i")
#     a.fromfile(fid, 1)

#     header = a[0]
#     footer = a[-1]
#     floats = np.array(np.frombuffer(fid.read(), dtype=np.float32))
#     floats = floats[:len(floats)-1]
#     floats = np.asarray(floats)
#     floats = np.reshape(floats, shape, order='F')
#     print("Filename is", filename)
#     print("Header =", header)
#     print("Footer =", footer)
#     if transpose:
#         return floats.T
#     return floats


# def write_dat(path, filename, arr):
#     floats = arr.flatten(order='F')

#     # Calculate header (number of total bytes in MDT)
#     header = np.array(arr.size * 4)

#     # Convert everything to bytes and write
#     floats = floats.tobytes()
#     header = header.tobytes()
#     footer = header
#     fid = open(os.path.join(path, filename), mode='wb')
#     fid.write(header)
#     fid.write(floats)
#     fid.write(footer)
#     fid.close()


IIin = 1440
JJin = 720

path1 = '.\\fortran\\data\\'

gmdt = rd.read_dat('shmdtout.dat', path=path1, shape=(IIin, JJin))

rd.write_dat('readwritetest.dat', gmdt, path=path1)

read_test = rd.read_dat('readwritetest.dat', path=path1, shape=(IIin, JJin))
print(gmdt[0, 0], read_test[0, 0])
print(gmdt.dtype, read_test.dtype)
print(gmdt.shape, read_test.shape)

print((gmdt == read_test))

print(np.all(gmdt == read_test))
# print(np.all(gmdt == read_test.T))
print(gmdt[-3, 0], read_test[-3, 0])

fig, (ax1, ax2) = plt.subplots(1, 2)
gmdt[gmdt < -1.8e+19] = -1.5
gmdt[gmdt > 1.5] = 1.5
read_test[read_test < -1.8e+19] = -1.5
read_test[read_test > 1.5] = 1.5
ax1.imshow(gmdt)
ax2.imshow(read_test)
plt.show()
