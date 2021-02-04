import numpy as np
from read_data import read_surface, read_surfaces, write_surface, apply_mask
import matplotlib.pyplot as plt
import cv2

# def scale_array(x, new_size):
#     min_el = np.min(x)
#     max_el = np.max(x)
#     numpy.array(Image.fromarray(arr).resize())
#     y = y / 255 * (max_el - min_el) + min_el
#     return y


def main():
    path1 = 'fortran/data/src/'
    mask = read_surface('mask_rr0008.dat', path1)
    mask = np.rot90(mask, 1)
    print(mask.shape)
    plt.imshow(mask)
    plt.show()

    resized_mask = cv2.resize(mask, (10801, 5400))
    resized_mask = np.round(resized_mask)
    # (5400, 10801)

    plt.imshow(resized_mask)
    plt.show()

    write_surface('mask_rr0030.dat', resized_mask, path1)

if __name__ == "__main__":
    main()