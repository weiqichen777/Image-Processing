import numpy as np

def GetMeanFilter():
    filter = np.zeros((3, 3))
    filter = filter + 1
    filter = filter / filter.sum()

    return filter

def GetGaussianFilter(sigma):
    x, y = np.mgrid[-1:2, -1:2]
    gaussion_filter = np.exp(-(x**2 + y**2) / 2 * pow(sigma, 2))
    gaussion_filter = gaussion_filter / gaussion_filter.sum() # normalize

    return gaussion_filter

def Convolution(dst, row, col, img_extend, filter):
    filter_row, filter_col = filter.shape

    RGB = np.array([0, 0, 0])
    for m in range(row):
        for n in range(col):
            for i in range(filter_row):
                for j in range(filter_col):
                    RGB = RGB + filter[i][j] * img_extend[i + m][j + n]
            dst[m][n] = RGB
            RGB = np.array([0, 0, 0])

    return dst

def GetMedianFilter(dst, row, col, img_extend):
    for i in range(row):
        for j in range(col):
            tmp = img_extend[i:i+3, j:j+3]
            tmp = tmp.reshape(9, 3)
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