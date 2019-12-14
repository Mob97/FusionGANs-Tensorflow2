import cv2
import tensorflow as tf
import os
import numpy as np
from net import Generator
import time
from preprocess import imsave
# def prepare_data2(dataset):
#     data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
#     data = glob.glob(os.path.join(data_dir, "*.jpg"))
#     data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
#     data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
#     return data

# def input_setup2(index):
#     padding=6
#     sub_ir_sequence = []
#     sub_vi_sequence = []
#     print(data_ir[index])
#     input_ir=(imread(data_ir[index])-127.5)/127.5
#     input_ir=np.pad(input_ir,((padding,padding),(padding,padding)),'edge')
#     w,h=input_ir.shape
#     input_ir=input_ir.reshape([w,h,1])
#     input_vi=(imread(data_vi[index])-127.5)/127.5
#     input_vi=np.pad(input_vi,((padding,padding),(padding,padding)),'edge')
#     w,h=input_vi.shape
#     input_vi=input_vi.reshape([w,h,1])
#     sub_ir_sequence.append(input_ir)
#     sub_vi_sequence.append(input_vi)
#     train_data_ir= np.asarray(sub_ir_sequence)
#     train_data_vi= np.asarray(sub_vi_sequence)
#     return train_data_ir,train_data_vi

def preprocessing(ir_img, vi_img):
    padding=6    
    
    # ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2YCrCb)
    ir_img = ir_img[:, :, 0]
    ir_img = (ir_img - 127.5)/127.5
    ir_img = np.pad(ir_img,((padding,padding),(padding,padding)),'edge')[None, ..., None]

    # vi_img = cv2.cvtColor(vi_img, cv2.COLOR_BGR2YCrCb)
    vi_img = vi_img[:, :, 0]
    vi_img = (vi_img - 127.5)/127.5    
    vi_img = np.pad(vi_img,((padding,padding),(padding,padding)),'edge')[None, ..., None]
    return ir_img, vi_img
# pir = '/home/minhbq/Downloads/INO_MainEntrance/INO_MainEntrance/INO_MainEntrance_T.avi'
# pvi = '/home/minhbq/Downloads/INO_MainEntrance/INO_MainEntrance/INO_MainEntrance_RGB.avi'

# pir = '/home/minhbq/Downloads/INO_MulitpleDeposit/INO_MulitpleDeposit_T.avi'
# pvi = '/home/minhbq/Downloads/INO_MulitpleDeposit/INO_MulitpleDeposit_RGB.avi'

pir = '/hdd/Minhbq/INO_TreesAndRunner/INO_TreesAndRunner/INO_TreesAndRunner_T.avi'
pvi = '/hdd/Minhbq/INO_TreesAndRunner/INO_TreesAndRunner/INO_TreesAndRunner_RGB.avi'

# pir = '/home/minhbq/Downloads/INO_ParkingEvening/INO_ParkingEvening/INO_ParkingEvening_T.avi'
# pvi = '/home/minhbq/Downloads/INO_ParkingEvening/INO_ParkingEvening/INO_ParkingEvening_RGB.avi'

# pri = '/home/minhbq/Downloads/INO_GroupFight/INO_GroupFight_T.avi'
# pvi = '/home/minhbq/Downloads/INO_GroupFight/INO_GroupFight_RGB.avi'

# pri = '/home/minhbq/Downloads/INO_ParkingEvening/INO_ParkingEvening/INO_ParkingEvening_T.avi'
# pvi = '/home/minhbq/Downloads/INO_ParkingEvening/INO_ParkingEvening/INO_ParkingEvening_RGB.avi'

cap1 = cv2.VideoCapture(pir)
cap2 = cv2.VideoCapture(pvi)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 5GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

print(tf.__version__)
generator = Generator()
generator.load_weights('./weights/generator/my_checkpoint9')
i = 0
while(cap1.isOpened() and cap2.isOpened()):
    # for _ in range(10):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 and not ret2:
        cap1.release()
        cap2.release()
        break
    f1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)
    f2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YCrCb)
    ir, vi = preprocessing(f1, f2)
    g_input = np.concatenate([ir, vi], axis=-1) 
    result = generator(g_input)
    result = result*127.5 + 127.5
    result = np.squeeze(result.numpy()).astype(np.uint8)
    gray = cv2.merge((result,result,result))
    fused_image = np.zeros_like(frame1)
    fused_image[:, :, 0] = result
    fused_image[:, :, 1] = f2[:, :, 1]
    fused_image[:, :, 2] = f2[:, :, 2] 
    fused_image = cv2.cvtColor(fused_image, cv2.COLOR_YCrCb2BGR)
    frame = np.concatenate((frame1, frame2, gray, fused_image), axis=1)
    cv2.imshow('out', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap1.release()
cap2.release()
cv2.destroyAllWindows()
