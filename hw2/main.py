import numpy as np
import sys
import cv2

from matplotlib import pyplot as plt
from intensity_transform import *
from noise_reduction import *
from bilateral_filter import *
from canny_edge import *
from edge_detect_array import *

path = sys.argv[1]
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
row, col= img.shape

dst = img.copy()
# dst = dst[:,:,::-1] # bgr to rgb
# dst = np.where(dst > np.mean(dst), 255, 0)

# === noise_reduction === 
# mean_filter = GetMeanFilter()
gaussian_filter = GetGaussianFilter(1)
dst = Convolution(dst, row, col, gaussian_filter)
dst_y = Convolution(dst, row, col, sobel)
dst_x = Convolution(dst, row, col, sobel.T)
dst = np.hypot(dst_y, dst_x)
dst = dst / dst.max() * 255

# === intensity_transform ===
dst = ConstrastAdjust(dst, 3, 3)
# dst = GammaCorrection(dst, 2.2)
# dst = LogTransform(dst)
# pixelVal_vec = np.vectorize(pixelVal)
# dst = pixelVal_vec(dst, 70, 30, 180, 230)
# dst = HistogramEqual(dst, row, col)

# === Canny edge detection ===
theta = np.arctan2(dst_y, dst_x)
dst = non_max_suppression(dst, theta)

# dst = ConstrastAdjust(dst, 3, 3)

dst = threshold(dst, 0.1, 0.6)

# === intensity_transform ===
dst = ConstrastAdjust(dst, 3, 3)
# dst = GammaCorrection(dst, 2.2)
# dst = LogTransform(dst)
# pixelVal_vec = np.vectorize(pixelVal)
# dst = pixelVal_vec(dst, 70, 30, 180, 230)
# dst = HistogramEqual(dst, row, col)

# dst = Convolution(dst, row, col, LoG3)


dst = np.round(dst)
dst = dst.astype(np.uint8)

# cv2.imshow('img', img)
# cv2.imshow('dst', dst)

# cv2.waitKey(0)

cv2.imwrite("images/test.png", dst)
# cv2.destroyAllWindows()

plt.figure(figsize=(10, 8))
plt.imshow(dst, cmap="gray")
plt.show()