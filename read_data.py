import os
import numpy as np
from utils import define_dims


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


def read_surface(filename, resolution=None, path=None, fortran=True,
                 nans=False, transpose=True):
    r"""Reshapes surface from 1d array into an array of
    (JJ, II) records.

    Ignores the header and footer of each record.

    Args:
        file (np.array): A .dat file containing a 1D array of floats
            respresenting input surface.

    Returns:
        np.array: data of size (II, JJ)
    """
    if resolution is None:
        resolution = parse_res(filename)
    
    II, JJ = (resolution)

    if path is None:
        path = ""

    filepath = os.path.join(os.path.normpath(path), filename)
    fid = open(filepath, mode='rb')

    # Loads Fortran array (CxR) or Python array (RxC)
    if fortran:
        floats = np.asfortranarray(np.frombuffer(fid.read(), dtype=np.float32))
        # Ignores the header and footer
        floats = floats[1:len(floats)-1]
        floats = np.array(floats)
        floats = np.reshape(floats, (II, JJ), order='F')

    else:
        floats = np.frombuffer(fid.read(), dtype=np.float32)
        floats = floats[1:len(floats)-1]
        floats = np.asarray(floats)
        floats = np.reshape(floats, (II, JJ))

    if nans:
        floats[floats <= -1.7e7] = np.nan
    if transpose:
        return floats.T

    return floats


def write_surface(filename, arr, path=None, fortran=False, nan_mask=None,
                  overwrite=False):
    r"""
    """
    if path is None:
        path = ""
    filepath = os.path.join(path, filename)

    if os.path.exists(filepath) and not overwrite:
        raise OSError("File already exists. Pass overwrite=True to overwrite.")

    if filepath[len(filepath)-4:] != '.dat':
        filepath += '.dat'

    if fortran:
        floats = arr.flatten(order='F')
    else:
        floats = arr.flatten()

    if nan_mask is not None:
        floats = floats * nan_mask
    floats[np.isnan(floats)] = -1.9e+19

    # Calculate header (number of total bytes in MDT)
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


def calc_residual(arr1, arr2):
    r""" Calculates the residual between two surfaces.

    Checks whether input arrays have the same dimensions.

    Args:
        arr1 (np.array): surface 1.
        arr2 (np.array): surface 2.

    Returns:
        np.array: An array representing the residual surface
            i.e. the difference between the input surfaces.
    """
    if np.shape(arr1) == np.shape(arr2):
        return np.subtract(arr1, arr2)
    else:
        return print("Cannot compute residual: surfaces are not same shape")


# def batch_reshape():
#     for filename in filenames:
#         reshape_data(filename, parse_res(filename), mdt=)


def main():
    print("read_data.py main")


if __name__ == '__main__':
    main()
