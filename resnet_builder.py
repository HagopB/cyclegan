from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from layers import *
from keras.regularizers import l2
from keras import backend as k
from resnet50 import ResNet50

def definenetFeat(input_shape, **kwargs):
    resnet50 = get_ResNet50(input_shape, **kwargs)
    return resnet50


def get_ResNet50(input_shape, trainable=False, pop=True, **kwargs):

    #importing convolutional layers of ResNet50 from keras
    model = ResNet50(include_top=False, weights='imagenet',input_shape=input_shape)
    if pop == True:
        model.layers.pop() # pop pooling layer
        model.layers.pop() # pop last activation layer

    #setting the convolutional layers to non-trainable 
    for layer in model.layers:
        layer.trainable = trainable
    
    print('Resnet50 for Perception loss:')
    model.summary()
    return(model)