import tensorflow as tf
from tensorflow.keras import Model
# from utils import *
import time

class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)
    """

    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`SpectralNormalization` must wrap a layer that'
                    ' contains a `kernel` for weights')

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_weight(
                shape=tuple([1, self.w_shape[-1]]),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                name='sn_u',
                trainable=False,
                dtype=tf.float32)

        super(SpectralNormalization, self).build()

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        """Call `Layer`"""
        # Recompute weights for each forward pass
        self._compute_weights()
        output = self.layer(inputs)
        return output

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        eps = 1e-12
        _u = tf.identity(self.u)
        _v = tf.matmul(_u, tf.transpose(w_reshaped))
        _v = _v / tf.maximum(tf.reduce_sum(_v**2)**0.5, eps)
        _u = tf.matmul(_v, w_reshaped)
        _u = _u / tf.maximum(tf.reduce_sum(_u**2)**0.5, eps)

        self.u.assign(_u)
        sigma = tf.matmul(tf.matmul(_v, w_reshaped), tf.transpose(_u))

        self.layer.kernel = self.w / sigma

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())
    
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__(name='generator')
        self.conv2d_1 = SpectralNormalization(tf.keras.layers.Conv2D(filters=256, 
                                                kernel_size=(5, 5), 
                                                padding='valid', 
                                                use_bias=True,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.0)
                                            ))
        self.conv2d_2 = SpectralNormalization(tf.keras.layers.Conv2D(filters=128, 
                                                kernel_size=(5, 5), 
                                                padding='valid', 
                                                use_bias=True,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.0)
                                            ))

        self.conv2d_3 = SpectralNormalization(tf.keras.layers.Conv2D(filters=64, 
                                                kernel_size=(3, 3), 
                                                padding='valid', 
                                                use_bias=True,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.0)
                                            ))
        self.conv2d_4 = SpectralNormalization(tf.keras.layers.Conv2D(filters=32, 
                                                kernel_size=(3, 3), 
                                                padding='valid', 
                                                use_bias=True,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.0)
                                            ))
        self.conv2d_5 = SpectralNormalization(tf.keras.layers.Conv2D(filters=1, 
                                                kernel_size=(1, 1), 
                                                padding='valid', 
                                                use_bias=True,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.0)
                                            ))
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_3 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_4 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        
        
    @tf.function(experimental_relax_shapes=True)
    def call(self, input_tensor, training=False):
        x = self.conv2d_1(input_tensor)
        x = self.bn_1(x, training=training)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x, training=training)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = self.conv2d_3(x)
        x = self.bn_3(x, training=training)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = self.conv2d_4(x)
        x = self.bn_4(x, training=training)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        x = self.conv2d_5(x)
        x = tf.keras.activations.tanh(x)

        return x
    
    def model(self):
        x = tf.keras.Input(shape=(132, 132, 2))
        return Model(inputs=[x], outputs=self.call(x))

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__(name='discriminator')
        self.conv2d_1 = SpectralNormalization(tf.keras.layers.Conv2D(filters=32, 
                                                kernel_size=(3, 3), 
                                                padding='valid', 
                                                strides = (2, 2),
                                                use_bias=True,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.0)
                                            ))

        self.conv2d_2 = SpectralNormalization(tf.keras.layers.Conv2D(filters=64, 
                                                kernel_size=(3, 3), 
                                                padding='valid', 
                                                strides = (2, 2),
                                                use_bias=True,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.0)
                                            ))

        self.conv2d_3 = SpectralNormalization(tf.keras.layers.Conv2D(filters=128, 
                                                kernel_size=(3, 3), 
                                                padding='valid',
                                                strides =  (2, 2), 
                                                use_bias=True,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.0)
                                            ))

        self.conv2d_4 = SpectralNormalization(tf.keras.layers.Conv2D(filters=256, 
                                                kernel_size=(3, 3), 
                                                padding='valid', 
                                                strides = (2, 2),
                                                use_bias=True,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.0)
                                            ))

        self.dense = tf.keras.layers.Dense(units=1,
                                          use_bias=True,
                                          kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                                          bias_initializer=tf.keras.initializers.Constant(value=0.0)
                                         )

        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_3 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_4 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

    @tf.function
    def call(self, input_tensor, training=True):
        x = self.conv2d_1(input_tensor)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x, training=training)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = self.conv2d_3(x)
        x = self.bn_3(x, training=training)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = self.conv2d_4(x)
        x = self.bn_4(x, training=training)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Flatten()(x)
        x = self.dense(x)
        return x
    
    def model(self):
        x = tf.keras.Input(shape=(120, 120, 1))
        return Model(inputs=[x], outputs=self.call(x))


    

    
