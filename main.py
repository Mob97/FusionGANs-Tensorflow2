from net import Generator, Discriminator
import tensorflow as tf
from preprocess import *
import time
import sys

print(tf.__version__)

ir_images_dir = 'Train_ir/' 
vi_images_dir = 'Train_vi/'
image_size = 132
label_size = 120
batch_size = 32
stride = 14
eps = 8.0
lda = 100.0
epoch = 30
lr = 1e-4


@tf.function
def gradient(img):
    laplace_filter = tf.constant([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], shape=[3, 3, 1, 1], dtype=tf.float32)    
    return tf.nn.conv2d(img, laplace_filter, strides=[1,1,1,1], padding='SAME')

@tf.function
def D_loss(d_fake, d_real):
    real_loss = tf.reduce_mean(tf.square(d_real-tf.random.uniform(shape=[batch_size,1], minval=0.7, maxval=1.2, dtype=tf.float32)))
    fake_loss = tf.reduce_mean(tf.square(d_fake-tf.random.uniform(shape=[batch_size,1], minval=0, maxval=0.3, dtype=tf.float32)))
    return real_loss + fake_loss
  
@tf.function  
def G_loss(d_fake, i_fake, i_ir, i_vi):
    v_gan = tf.reduce_mean(tf.square(d_fake - tf.random.uniform(shape=[batch_size, 1], minval=0.7, maxval=1.2, dtype=tf.float32)))
    content = tf.reduce_mean(tf.square(i_fake - i_ir)) + eps*tf.reduce_mean(tf.square(gradient(i_fake) - gradient(i_vi)))
    return v_gan + lda*content

@tf.function()
def training_step(generator, discriminator, d_op, g_op, images_ir, images_vi, labels_ir, labels_vi, k = 2):
    g_input = tf.concat([images_ir, images_vi], axis=-1)    
    with tf.GradientTape() as g_t:
        i_fakes = generator(g_input, True)
        for i in range(k):        
            with tf.GradientTape() as d_t:
                d_fakes = discriminator(i_fakes)
                d_reals = discriminator(labels_vi)
                d_loss = D_loss(d_fakes, d_reals)
            d_gradients = d_t.gradient(d_loss, discriminator.trainable_variables)
            d_op.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        g_loss = G_loss(d_fakes, i_fakes, labels_ir, labels_vi)
        g_gradients = g_t.gradient(g_loss, generator.trainable_variables)
        g_op.apply_gradients(zip(g_gradients, generator.trainable_variables))
    return d_loss, g_loss

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 5GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
        print(e)


train_data_ir, train_label_ir = get_images(ir_images_dir, image_size, label_size, stride)
train_data_vi, train_label_vi = get_images(vi_images_dir, image_size, label_size, stride)

print(train_data_ir.shape)
print(train_data_vi.shape)
print(train_label_ir.shape)
print(train_label_vi.shape)

generator = Generator()
discriminator = Discriminator()
previos = -1
# generator.load_weights('./save-eps5/generator/my_checkpoint{}'.format(previos))
# discriminator.load_weights('./save-eps5/discriminator/my_checkpoint{}'.format(previos))
start_time = time.time()
counter = 0
d_op = tf.keras.optimizers.Adam(lr)
g_op = tf.keras.optimizers.Adam(lr)

for i in range(previos + 1, previos + 1 + epoch):        
    batch_idxs = len(train_data_ir) // batch_size
    for idx in range(0, batch_idxs):
        batch_images_ir = train_data_ir[idx*batch_size : (idx+1)*batch_size]
        batch_labels_ir = train_label_ir[idx*batch_size : (idx+1)*batch_size]
        batch_images_vi = train_data_vi[idx*batch_size : (idx+1)*batch_size]
        batch_labels_vi = train_label_vi[idx*batch_size : (idx+1)*batch_size]         
        counter += 1
        d_loss, g_loss = training_step(generator, discriminator, d_op, g_op, batch_images_ir, batch_images_vi, batch_labels_ir, batch_labels_vi, 2)
        if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_d: [%.8f],loss_g:[%.8f]" \
            % ((i), counter, time.time()-start_time, d_loss, g_loss))        
    generator.save_weights('./save-eps5/generator/my_checkpoint{}'.format(i))
    discriminator.save_weights('./save-eps5/discriminator/my_checkpoint{}'.format(i))
