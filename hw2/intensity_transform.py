import numpy as np

def ConstrastAdjust(dst, contrast, brightness):
    a = contrast
    b = brightness

    dst = dst * a + b
    dst[dst > 255] = 255
    dst[dst < 0] = 0

    return dst

def GammaCorrection(dst, gamma):
    dst = (dst / 255) ** (1 / gamma)
    dst = dst * 255
    dst[dst > 255] = 255
    dst[dst < 0] = 0

    return dst

def LogTransform(dst):
    c = 255/(np.log(1 + np.max(dst))) 
    dst = c * np.log(1 + dst)

    return dst

def pixelVal(dst, r1, s1, r2, s2): 
    if (0 <= dst and dst <= r1): 
        return (s1 / r1)*dst 
    elif (r1 < dst and dst <= r2): 
        return ((s2 - s1)/(r2 - r1)) * (dst - r1) + s1 
    else: 
        return ((255 - s2)/(255 - r2)) * (dst - r2) + s2 

def HistogramEqual(dst, row, col):
    P = dst

    P_cdf = np.zeros((256,))
    P_sum = 0
    for i in range(256):
        P_count = np.count_nonzero(P == i)
        if P_count != 0:
            P_sum = P_sum + P_count
            P_cdf[i] = P_sum

    P_eq = P_cdf / (row * col * 255)

    for i in range(row):
        for j in range(col):
            P_color = dst[i][j]
            dst[i][j] = P_eq[P_color]

    return dst