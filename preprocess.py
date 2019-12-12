
import numpy as np
import os
# from PIL import Image
<<<<<<< HEAD
# import scipy.misc
=======
#import scipy.misc
>>>>>>> af69680e9f2bb9ab50f2062bd05d5280de767774
import glob
import cv2

def get_images(data_dir, image_size, label_size, stride):
    data = prepare_data(data_dir)
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(image_size - label_size) // 2
    for i in range(len(data)):
        input_ = (imread(data[i])-127.5)/127.5
        height, width = input_.shape[:2]
        for x in range(0, height - image_size + 1, stride):
            for y in range(0, width - image_size + 1, stride):
                sub_input = input_[x:x+image_size, y:y+image_size].reshape([image_size, image_size, 1])  
                sub_label = input_[x+padding:x+padding+label_size, y+padding:y+padding+label_size].reshape([label_size, label_size, 1])
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
    sub_input_sequence = np.asarray(sub_input_sequence, dtype=np.float32) 
    sub_label_sequence = np.asarray(sub_label_sequence, dtype=np.float32)
    return sub_input_sequence, sub_label_sequence

def prepare_data(data_path):
    """
    Args:
      data_path: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    data_path = os.path.join(os.getcwd(), data_path)
    images_path = glob.glob(os.path.join(data_path, "*.bmp"))
    images_path.extend(glob.glob(os.path.join(data_path, "*.tif")))
    images_path.sort(key=lambda x: int(x[len(data_path) :-4]))
    return images_path


def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img[:, :, 0]

def imsave(image, path):
    return scipy.misc.imsave(path, image)

# def imread(path):
#     return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float32)
