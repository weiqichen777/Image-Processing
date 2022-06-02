import numpy as np

def distance(x, y, i, j):
    return np.sqrt(((x-i)**2 + (y-j)**2))

def gaussian(x, sigma):
    return np.exp(- (x ** 2) / (2 * sigma ** 2))

def BilateralFilter(dst, row, col, img_extend, sigma_d=10, sigma_r=10):
    for x in range(row):
        for y in range(col):
            pixel_filter = np.zeros((3,))
            w_sum = np.zeros((3,))
            for i in range(x, x+3):
                for j in range(y, y+3):
                    d = distance(i, j, x+1, y+1)
                    Id = img_extend[i][j] - img_extend[x+1][y+1]
                    space_w = gaussian(d, sigma_d)
                    color_w = gaussian(Id, sigma_r)
                    w = space_w * color_w
                    pixel_filter += img_extend[i][j] * w
                    w_sum += w
            pixel_filter = pixel_filter / w_sum
            dst[x][y] = pixel_filter
    
    return dst