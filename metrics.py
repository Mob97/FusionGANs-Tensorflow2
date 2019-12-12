from skimage.measure import compare_ssim
import numpy as np
from skimage.measure import shannon_entropy

def cross_covariance(x, y, mu_x, mu_y):
    return 1/(x.size-1)*np.sum((x-mu_x)*(y-mu_y))

def cal_ssim(x, y):
    L = np.max(np.array([x, y])) - np.min(np.array([x, y]))
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sig_x = np.std(x)
    sig_y = np.std(y)
    sig_xy = cross_covariance(x, y, mu_x, mu_y)
    C1 = (0.01*L)**2
    C2 = (0.03*L)**2
    C3 = C2/2
    return (2*mu_x*mu_y + C1)*(2*sig_x*sig_y + C2)*(sig_xy + C3)/((mu_x**2+mu_y**2 + C1)*(sig_x**2 + sig_y**2 + C2)*(sig_x*sig_y + C3))


def ssim(ir, vi, fused):
    ssim_af = cal_ssim(ir, fused)
    ssim_bf = cal_ssim(vi, fused)
    return ssim_af + ssim_bf

def entropy(img):
    return shannon_entropy(img)

def standard_deviation(img):
    return np.std(img)

def correlaton_coefficients(ir, vi, fused):
    rx = np.corrcoef(ir.flat, fused.flat)
    ry = np.corrcoef(vi.flat, fused.flat)
    return (rx[0,1]+ry[0,1])/2

def spatialFrequency(I):
    RF = np.sqrt(np.mean(np.square(I[:, 1:] - I[:, :-1])))
    CF = np.sqrt(np.mean(np.square(I[1:, :] - I[:-1, :])))
<<<<<<< HEAD
    return np.sqrt(RF + CF)
=======
    return np.sqrt(RF**2 + CF**2)
>>>>>>> af69680e9f2bb9ab50f2062bd05d5280de767774
