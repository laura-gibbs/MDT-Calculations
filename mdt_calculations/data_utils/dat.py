import os
import numpy as np
import math


def parse_res(filename):
    for letter in range(len(filename)):
        if filename[letter:letter+2] == 'rr':
            res = 1/int(filename[letter+2:letter+6])
            return res


def parse_mdt(filename):
    r"""Checks whether input file is MDT by counting number of underscores.

    Args:
        filename (String)

    Returns:
        boolean

    """
    if filename.count("_") == 3:
        return True
    else:
        return False


def read_surface(filename, path=None, fortran=True, nans=True,
                 transpose=True):
    r"""Reshapes surface from 1d array into an array of
    (II, JJ) records.

    Ignores the header and footer of each record.

    Args:
        file (np.array): A .dat file containing a 1D array of floats
            respresenting input surface.

    Returns:
        np.array: data of size (II, JJ)
    """
    order = 'F' if fortran else 'C'

    if path is None:
        path = ""

    filepath = os.path.join(os.path.normpath(path), filename)
    fid = open(filepath, mode='rb')
    buffer = fid.read(4)
    size = np.frombuffer(buffer, dtype=np.int32)[0]
    shape = (int(math.sqrt(size//8)*2), int(math.sqrt(size//8)))
    fid.seek(0)

    # Loads Fortran array (CxR) or Python array (RxC)
    floats = np.array(np.frombuffer(fid.read(), dtype=np.float32), order=order)
    floats = floats[1:len(floats)-1]
    floats = np.reshape(floats, shape, order=order)

    if nans:
        floats[floats <= -1.7e7] = np.nan
    if transpose:
        return floats.T

    return floats


def read_surfaces(filename, path=None, fortran=True, nans=True,
                  transpose=False, number=1, start=None):
    r"""
    """
    order = 'F' if fortran else 'C'

    if path is None:
        path = ""

    arr = []
    filepath = os.path.join(os.path.normpath(path), filename)
    fid = open(filepath, mode='rb')
    buffer = fid.read(4)
    size = np.frombuffer(buffer, dtype=np.int32)[0]
    shape = (int(math.sqrt(size//8)*2), int(math.sqrt(size//8)))

    hdr_pointer = (shape[0]*shape[1]+2)*4
    if start is not None:
        fid.seek(start*hdr_pointer)

    # Loads Fortran array (CxR) or Python array (RxC)
    while buffer != b'' and len(arr) <= number-1:
        floats = np.array(np.frombuffer(fid.read(size),
                                        dtype=np.float32), order=order)
        floats = np.reshape(floats, shape, order=order)
        arr.append(floats)
        print(f'Loaded MDT #{(start+len(arr))}')
        footer_value = np.frombuffer(fid.read(4), dtype=np.int32)[0]
        buffer = fid.read(4)

    arr = np.array(arr)
    if nans:
        arr[arr <= -1.7e7] = np.nan
    if transpose:
        return np.transpose(arr, (0, 2, 1))

    return arr


def write_surface(filename, arr, path=None, fortran=False, nan_mask=None,
                  overwrite=False):
    r"""
    """
    order = 'F' if fortran else 'C'
 
    if path is None:
        path = ""
    filepath = os.path.join(path, filename)

    if os.path.exists(filepath) and not overwrite:
        raise OSError("File already exists. Pass overwrite=True to overwrite.")

    if filepath[len(filepath)-4:] != '.dat':
        filepath += '.dat'

    arr = arr.astype('float32')
    floats = arr.flatten(order=order)

    if nan_mask is not None:
        floats = floats * nan_mask
    floats[np.isnan(floats)] = -1.9e+19

    # Calculate header (number of total bytes in MDT)
    print('array size = ', floats.size)
    header = np.array(arr.size * 4)

    # Convert everything to bytes and write
    floats = floats.tobytes()
    header = header.tobytes()
    footer = header
    fid = open(filepath, mode='wb')
    fid.write(header)
    fid.write(floats)
    fid.write(footer)
    fid.close()


def main():
    print("read_data.py main")


if __name__ == '__main__':
    main()
