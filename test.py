from preprocess import *
from metrics import *

ir_path = '/home/minhbq/MyProjects/Thesis_FusionGAN/my_ir/47.bmp'
vi_path = '/home/minhbq/MyProjects/Thesis_FusionGAN/my_vi/47.bmp'
fu1_path = '/home/minhbq/MyProjects/Thesis_FusionGAN/result/_epoch9/F9_46.bmp'

# ir2_path = '/hdd/Minhbq/FusionGAN/my_ir/47.bmp'
# vi2_path = '/hdd/Minhbq/FusionGAN/my_vi/47.bmp'
# fu2_path = '/hdd/Minhbq/FusionGAN1/result/epoch3/F9_46.bmp'
fu2_path = '/hdd/Minhbq/FusionGAN1/result/epoch9_1/F9_46.bmp'

ir = imread(ir_path)
vi = imread(vi_path)
fu1 = imread(fu1_path)
fu2 = imread(fu2_path)
# ir = (ir - 127.5)/127.5
# vi = (vi - 127.5)/127.5
# fu1 = (fu1 - 127.5)/127.5
# fu2 = (fu2 - 127.5)/127.5
print('SSIM 1: {}, SSIM 2: {}'.format(ssim(ir, vi, fu1), ssim(ir, vi, fu2)))
print('Entropy 1: {}, Entropy 2: {}'.format(entropy(fu1), entropy(fu2)))
print('CC 1: {}, CC 2: {}'.format(correlaton_coefficients(ir, vi, fu1) , correlaton_coefficients(ir, vi, fu2)))
print('SD 1: {}, SD 2: {}'.format(standard_deviation(fu1), standard_deviation(fu2)))
print('SF 1: {}, SF 2: {}'.format(spatialFrequency(fu1), spatialFrequency(fu2)))