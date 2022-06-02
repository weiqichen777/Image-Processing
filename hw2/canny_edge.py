import numpy as np


def non_max_suppression(img, D):
    row, col = img.shape
    res = np.zeros((row, col))
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1, row-1):
        for j in range(1, col-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    res[i,j] = img[i,j]
                else:
                    res[i,j] = 0

            except IndexError as e:
                pass
    
    return res

def threshold(img, lowThresholdRatio, highThresholdRatio):
    
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    row, col = img.shape
    res = np.zeros((row, col))
    
    weak = 25
    strong = 255
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res