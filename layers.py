from keras.engine.topology import Layer
import keras.backend as K

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
    def __init__(self, **kwargs):
        super(InstanceNormalization2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[1],), initializer="one", trainable=True)
        self.shift = self.add_weight(name='shift', shape=(input_shape[1],), initializer="zero", trainable=True)
        super(InstanceNormalization2D, self).build(input_shape)

    def call(self, x, mask=None):
        def image_expand(tensor):
            return K.expand_dims(K.expand_dims(tensor, -1), -1)

        def batch_image_expand(tensor):
            return image_expand(K.expand_dims(tensor, 0))

        hw = K.cast(x.shape[1] * x.shape[2], K.floatx())
        mu = K.sum(x, [-1, -2]) / hw
        mu_vec = image_expand(mu)
        sig2 = K.sum(K.square(x - mu_vec), [-1, -2]) / hw
        y = (x - mu_vec) / (K.sqrt(image_expand(sig2)) + K.epsilon())

        scale = batch_image_expand(self.scale)
        shift = batch_image_expand(self.shift)
        return scale*y + shift

    def compute_output_shape(self, input_shape):
        return input_shape
