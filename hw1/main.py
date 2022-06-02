import numpy as np
import sys
import cv2

from intensity_transform import *
from noise_reduction import *
from bilateral_filter import *

path = sys.argv[1]
img = cv2.imread(path)
row, col, channel = img.shape

dst = img.copy()

### intensity_transform
#dst = ConstrastAdjust(dst, 0.5, 50)
dst = GammaCorrection(dst, 0.5)
#dst = LogTransform(dst)
#pixelVal_vec = np.vectorize(pixelVal)
#dst = pixelVal_vec(dst, 70, 30, 180, 230)
#dst = HistogramEqual(dst, row, col)

dst_extend = np.zeros((row + 2, col + 2, channel))

for i in range(1, row + 1):
    for j in range(1, col + 1):
        dst_extend[i][j] = dst[i-1][j-1]

### noise_reduction
#mean_filter = GetMeanFilter()
#dst = Convolution(dst, row, col, dst_extend, mean_filter)
#dst = GetMedianFilter(dst, row, col, dst_extend)
#dst = GetAlphaTrimmedMeanFilter(dst, row, col, dst_extend, 6)

### bilateral_filter
#dst = BilateralFilter(dst, row, col, dst_extend, 1, 5)


dst = np.round(dst)
dst = dst.astype(np.uint8)

cv2.imshow('img', img)
cv2.imshow('dst', dst)

cv2.waitKey(0)

cv2.imwrite("images/test.png", dst)
cv2.destroyAllWindows()