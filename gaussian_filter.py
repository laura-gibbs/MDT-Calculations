from scipy.ndimage import gaussian_filter
import numpy as np


def apply_gaussian(arr, sigma=3, bd=-1.5):
    V = arr.copy()
    mask = np.ones_like(arr)
    mask[np.isnan(arr)] = np.nan
    V[np.isnan(arr)] = 0
    VV = gaussian_filter(V, sigma=sigma)

    W = 0*arr.copy() + 1
    W[np.isnan(arr)] = 0
    WW = gaussian_filter(W, sigma=sigma)

    arr = VV/WW * mask
    # arr[np.isnan(arr)] = bd

    return arr


def main():
    pass


if __name__ == "__main__":
    main()