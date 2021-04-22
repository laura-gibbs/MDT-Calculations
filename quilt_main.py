import numpy as np
import math
from skimage import io, util
import heapq
from skimage.transform import rescale, resize
from PIL import Image
import argparse
import os
import glob


def randomPatch(texture, block_size):
    h, w, _ = texture.shape
    i = np.random.randint(h - block_size)
    j = np.random.randint(w - block_size)

    return texture[i:i+block_size, j:j+block_size]

def L2OverlapDiff(patch, block_size, overlap, res, y, x):
    error = 0
    if x > 0:
        left = patch[:, :overlap] - res[y:y+block_size, x:x+overlap]
        error += np.sum(left**2)

    if y > 0:
        up   = patch[:overlap, :] - res[y:y+overlap, x:x+block_size]
        error += np.sum(up**2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
        error -= np.sum(corner**2)

    return error
 

# def randomBestPatch(texture, block_size, overlap, res, y, x):
#     h, w, _ = texture.shape
#     errors = np.zeros((h - block_size, w - block_size))

#     for i in range(h - block_size):
#         for j in range(w - block_size):
#             patch = texture[i:i+block_size, j:j+block_size]
#             e = L2OverlapDiff(patch, block_size, overlap, res, y, x)
#             errors[i, j] = e

#     i, j = np.unravel_index(np.argmin(errors), errors.shape)
#     return texture[i:i+block_size, j:j+block_size]


def randomBestPatch(textures, block_size, overlap, res, y, x):
    h, w, c = textures[0].shape
    errors = np.zeros((block_size // 2, block_size // 2))

    random_textures = np.empty((block_size // 2, block_size // 2, block_size, block_size, c))
    for i in range(block_size // 2):
        for j in range(block_size // 2):
            patch = textures[np.random.randint(0, len(textures)), :block_size, :block_size]
            # patch = patch[i:i+block_size, j:j+block_size]
            e = L2OverlapDiff(patch, block_size, overlap, res, y, x)
            errors[i, j] = e
            random_textures[i, j] = patch

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return random_textures[i, j]


def minCutPath(errors):
    # dijkstra's algorithm vertical
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)

    h, w = errors.shape
    seen = set()

    while pq:
        error, path = heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[-1]

        if curDepth == h:
            return path

        for delta in -1, 0, 1:
            nextIndex = curIndex + delta

            if 0 <= nextIndex < w:
                if (curDepth, nextIndex) not in seen:
                    cumError = error + errors[curDepth, nextIndex]
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))

                    
def minCutPatch(patch, block_size, overlap, res, y, x):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = np.sum(left**2, axis=2)
        for i, j in enumerate(minCutPath(leftL2)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)

    return patch


def quilt(image_path, block_size, num_block, mode, sequence=False):
    textures = np.stack([
        util.img_as_float32(Image.open(path)) for path in glob.glob(os.path.join(image_path, '*.png'))
    ])

    if textures.ndim < 4:
        textures = np.expand_dims(textures, -1)

    overlap = block_size // 2
    num_blockHigh, num_blockWide = num_block

    h = (num_blockHigh * block_size) - (num_blockHigh - 1) * overlap
    w = (num_blockWide * block_size) - (num_blockWide - 1) * overlap
    c = textures[0].shape[2]

    res = np.zeros((h, w, c))

    for i in range(num_blockHigh):
        print(f'{i}/{num_blockHigh}')
        for j in range(num_blockWide):
            # print(i, j)
            # print(f'{i * num_blockHigh + j}/{num_blockHigh * num_blockWide}')
            y = i * (block_size - overlap)
            x = j * (block_size - overlap)

            if i == 0 and j == 0 or mode == "Random":
                patch = textures[np.random.randint(0, len(textures)), :block_size, :block_size]
            elif mode == "Best":
                patch = randomBestPatch(textures, block_size, overlap, res, y, x)
            elif mode == "Cut":
                patch = randomBestPatch(textures, block_size, overlap, res, y, x)
                patch = minCutPatch(patch, block_size, overlap, res, y, x)
            
            res[y:y+block_size, x:x+block_size] = patch

    print(res.shape, res.min(), res.max())
    if c == 1:
        res = res.squeeze(-1)
    image = Image.fromarray((res * 255).astype(np.uint8))
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, type=str, help="path to images")
    parser.add_argument("-b", "--block_size", type=int, default=32, help="block size in pixels")
    parser.add_argument("-n", "--num_block", type=int, default=10, help="number of blocks you want")
    parser.add_argument("-m", "--mode", type=str, default='Cut', help="which mode --random placement of block(Random)/Neighbouring blocks constrained by overlap(Best)/Minimum error boundary cut(Cut)")
    args = parser.parse_args()

    np.random.seed(0)
    path = args.path
    block_size = args.block_size
    num_block = args.num_block
    mode = args.mode
    image = quilt(path, block_size, (num_block, num_block), mode)
    image.save(f'{args.mode.lower()}-n{args.num_block}.png')