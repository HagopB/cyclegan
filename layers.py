from keras.engine.topology import Layer
import keras.backend as K
from keras.layers import InputSpec
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras import backend as k

'''
*****************************************************************************
**************************  ReflectPadding2D   ******************************
*****************************************************************************
'''
class ReflectPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = padding
        super(ReflectPadding2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReflectPadding2D, self).build(input_shape)

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            T = K.theano.tensor
            p0, p1 = self.padding[0], self.padding[1]
            y = T.zeros((x.shape[0], x.shape[1], x.shape[2]+(2*p0), x.shape[3]+(2*p1)), dtype=K.theano.config.floatX)
            y = T.set_subtensor(y[:, :, p0:-p0, p1:-p1], x)
            y = T.set_subtensor(y[:, :, :p0, p1:-p1], x[:, :, p0:0:-1, :])
            y = T.set_subtensor(y[:, :, -p0:, p1:-p1], x[:, :, -2:-2-p0:-1])
            y = T.set_subtensor(y[:, :, p0:-p0, :p1], x[:, :, :, p1:0:-1])
            y = T.set_subtensor(y[:, :, p0:-p0, -p1:], x[:, :, :, -2:-2-p1:-1])
            y = T.set_subtensor(y[:, :, :p0, :p1], x[:, :, p0:0:-1, p1:0:-1])
            y = T.set_subtensor(y[:, :, -p0:, :p1], x[:, :, -2:-2-p0:-1, p1:0:-1])
            y = T.set_subtensor(y[:, :, :p0, -p1:], x[:, :, p0:0:-1, -2:-2-p1:-1])
            y = T.set_subtensor(y[:, :, -p0:, -p1:], x[:, :, -2:-2-p0:-1, -2:-2-p1:-1])
        else:
            raise NotImplemented("Please complete `CycGAN/layers/padding.py` to run on backend {}.".format(K.backend()))
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]+(2*self.padding[0]), input_shape[3]+(2*self.padding[1]))


'''
*****************************************************************************
**************************  InstanceNormalization  **************************
*****************************************************************************
'''
class InstanceNormalization2D(Layer):
    def __init__(self,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 epsilon=1e-3,
                 **kwargs):
        super(InstanceNormalization2D, self).__init__(**kwargs)
        if K.image_data_format() is 'channels_first':
            self.axis = 1
        else: # image channels x.shape[3]
            self.axis = 3
        print()
        self.epsilon = epsilon
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[self.axis],),
                                     initializer=self.gamma_initializer,
                                     trainable=True,
                                     name='gamma')
        self.beta = self.add_weight(shape=(input_shape[self.axis],),
                                    initializer=self.beta_initializer,
                                    trainable=True,
                                    name='beta')
        super(InstanceNormalization2D, self).build(input_shape)

    def call(self, x):
        # spatial dimensions of input
        if K.image_data_format() is 'channels_first':
            x_w, x_h = (2, 3)
        else:
            x_w, x_h = (1, 2)

        # Very similar to batchnorm, but normalization over individual inputs.

        hw = K.cast(K.shape(x)[x_h]* K.shape(x)[x_w], K.floatx())

        # Instance means
        mu = K.sum(x, axis=x_w)
        mu = K.sum(mu, axis=x_h)
        mu = mu / hw
        mu = K.reshape(mu, (K.shape(mu)[0], K.shape(mu)[1], 1, 1))

        # Instance variences
        sig2 = K.square(x - mu)
        sig2 = K.sum(sig2, axis=x_w)
        sig2 = K.sum(sig2, axis=x_h)
        sig2 = K.reshape(sig2, (K.shape(sig2)[0], K.shape(sig2)[1], 1, 1))

        # Normalize
        y = (x - mu) / K.sqrt(sig2 + self.epsilon)

        # Scale and Shift
        if K.image_data_format() is 'channels_first':
            gamma = K.reshape(self.gamma, (1, K.shape(self.gamma)[0], 1, 1))
            beta = K.reshape(self.beta, (1, K.shape(self.beta)[0], 1, 1))
        else:
            gamma = K.reshape(self.gamma, (1, 1, 1, K.shape(self.gamma)[0]))
            beta = K.reshape(self.beta, (1, 1, 1, K.shape(self.beta)[0]))
        return gamma * y + beta
    
'''
*****************************************************************************
**************************  OTHER util layers   *****************************
*****************************************************************************
'''
conv_init = RandomNormal(0, 0.02) # for convolution kernel
gamma_init = RandomNormal(1., 0.02) # for batch normalization

def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer = conv_init, *a, **k)
def batchnorm():
    return BatchNormalization(momentum=0.9, axis=3, epsilon=1.01e-5, 
                              gamma_initializer = gamma_init)