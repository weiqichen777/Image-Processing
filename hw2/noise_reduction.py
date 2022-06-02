import numpy as np
from numba import jit

def GetMeanFilter():
    filter = np.zeros((3, 3))
    filter = filter + 1
    filter = filter / filter.sum()
    
    return filter

def GetGaussianFilter(sigma):
    x, y  = np.mgrid[-1:2, -1:2]
    gaussion_filter = np.exp(-(x**2 + y**2) / 2 * pow(sigma, 2))
    gaussion_filter = gaussion_filter / gaussion_filter.sum() # normalize

    return gaussion_filter

@jit
def Convolution(dst, row, col, ftr):
    tmp = np.zeros((row, col))

    for i in range(2, row-2):
        for j in range(2, col-2):
            for k in range(i-1, i+2):
                for l in range(j-1, j+2):
                    tmp[i][j] += dst[k][l] * ftr[k-i+1][l-j+1]

    return tmp


def GetMedianFilter(dst, row, col, img_extend):
    for i in range(row):
        for j in range(col):
            tmp = img_extend[i:i+3, j:j+3]
            # tmp = tmp.reshape(9, 3)
            median = np.median(tmp, axis=0)
            dst[i][j] = median

    return dst

def GetAlphaTrimmedMeanFilter(dst, row, col, img_extend, d):
    d2 = int(d/2)

    for i in range(row):
        for j in range(col):
            tmp = img_extend[i:i+3, j:j+3]
            tmp = tmp.reshape(9, 3)
            tmp = tmp[d2:9-d2, 0:3]
            mean = np.mean(tmp, axis=0)
            dst[i][j] = mean
    
    return dst