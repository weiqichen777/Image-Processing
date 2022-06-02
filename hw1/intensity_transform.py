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
    R = dst[:,:,0]
    G = dst[:,:,1]
    B = dst[:,:,2]

    R_cdf = np.zeros((256,))
    R_sum = 0
    G_cdf = np.zeros((256,))
    G_sum = 0
    B_cdf = np.zeros((256,))
    B_sum = 0
    for i in range(256):
        R_count = np.count_nonzero(R == i)
        if R_count != 0:
            R_sum = R_sum + R_count
            R_cdf[i] = R_sum

        G_count = np.count_nonzero(G == i)
        if G_count != 0:
            G_sum = G_sum + G_count
            G_cdf[i] = G_sum
        
        B_count = np.count_nonzero(B == i)
        if B_count != 0:
            B_sum = B_sum + B_count
            B_cdf[i] = B_sum

    R_eq = R_cdf / 480000 * 255
    G_eq = G_cdf / 480000 * 255
    B_eq = B_cdf / 480000 * 255
    
    for i in range(row):
        for j in range(col):
            R_color, G_color, B_color = dst[i][j]
            dst[i][j] = [R_eq[R_color], G_eq[G_color], B_eq[B_color]]
    
    return dst