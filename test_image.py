import os

import tensorflow as tf
import glob
import time
import numpy as np
import scipy.misc
import scipy.ndimage
#import matplotlib.pyplot as plt
from net import Generator, Discriminator
from preprocess import imread
import cv2
from metrics import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
tf.autograph.set_verbosity(0)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 5GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

# print(tf.__version__)


def prepare_data2(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.tif"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def input_setup2(index):
    padding=6
    sub_ir_sequence = []
    sub_vi_sequence = []
    _ir = imread(data_ir[index])
    _vi = imread(data_vi[index])
    input_ir=(_ir-127.5)/127.5
    input_ir=np.pad(input_ir,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(_vi-127.5)/127.5
    input_vi=np.pad(input_vi,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    return train_data_ir,train_data_vi

def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img[:, :, 0]

def imsave(image, path):
    return scipy.misc.imsave(path, image)

g = Generator()
g.load_weights('./weights/generator/my_checkpoint9')


data_ir=prepare_data2('Test_ir')
data_vi=prepare_data2('Test_vi')

g = Generator()
g.load_weights('./weights/generator/my_checkpoint9')
image_path = os.path.join(os.getcwd(), 'result','test2')
if not os.path.exists(image_path):
    os.makedirs(image_path)
for i in range(len(data_ir)):
    start=time.time()
    train_data_ir,train_data_vi=input_setup2(i)
    g_input = np.concatenate([train_data_ir, train_data_vi], axis=-1) 
#     print(g_input)
#     g_input = tf.concat([train_data_ir, train_data_vi], axis=-1) 
#     print(g_input)
    result = g(g_input)
#     result=result*127.5+127.5
#     result = result.squeeze()        
    save_path = os.path.join(image_path, str(i+1)+".bmp")
    end=time.time()
    
#     print(result[0][:,:,0])
#     break
    imsave(result[0][:,:,0], save_path)
    print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
