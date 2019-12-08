from skimage.metrics import structural_similarity
import numpy as np
from skimage.measure import shannon_entropy

def ssim(ir, vi, fused):
    ssim_af = structural_similarity(ir, fused)
    ssim_bf = structural_similarity(vi, fused)
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
    RF = np.sqrt(np.sum(np.square(I[:, 1:] - I[:, :-1])))
    CF = np.sqrt(np.sum(np.square(I[1:, :] - I[:-1, :])))
    return np.sqrt(RF**2 + CF**2)
