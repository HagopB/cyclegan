from keras.layers import UpSampling2D
from keras.layers import Add
from layers import *


'''
*****************************************************************************
********************************  defineG   *********************************
*****************************************************************************
'''
def defineG(which_model_netG, input_shape, output_shape, ngf, **kwargs):
    output_nc = output_shape[2]
    if which_model_netG == 'resnet_6blocks':
        return resnet_6blocks(input_shape, output_nc, ngf, **kwargs)
    elif which_model_netG == 'unet_128':
        return unet_128(input_shape, output_nc, ngf, **kwargs)
        

'''
*****************************************************************************
****************************  Generator: Resnet *****************************
*****************************************************************************
'''
padding = ZeroPadding2D # or use ReflectPadding2D

def normalize(**kwargs):
    return batchnorm()#axis=get_filter_dim()
    #return InstanceNormalization2D()

def scaleup(input, ngf, kss, strides, padding):
    x = UpSampling2D(strides)(input)
    x = Conv2D(ngf, kss, padding=padding, kernel_initializer = conv_init)(x)
    return x

def res_block(input, filters, kernel_size=(3,3), strides=(1,1)):
    x = padding()(input)
    x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides=strides,kernel_initializer = conv_init)(x)
    x = normalize()(x, training=1)
    x = Activation('relu')(x)

    x = padding()(x)
    x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides=strides, kernel_initializer = conv_init)(x)
    x = normalize()(x, training=1)

    merged = Add()([input, x])
    return merged

def resnet_6blocks(input_shape, output_nc, ngf, **kwargs):
    ks = 3
    f = 7
    p = int((f-1)/2)

    input = Input(input_shape)
    x = padding((p,p))(input)
    x = Conv2D(ngf, (f,f), kernel_initializer = conv_init)(x)
    x = normalize()(x, training=1)
    x = Activation('relu')(x)

    x = Conv2D(ngf*2, (ks,ks), strides=(2,2), padding='same', kernel_initializer = conv_init)(x)
    x = normalize()(x, training=1)
    x = Activation('relu')(x)

    x = Conv2D(ngf*4, (ks,ks), strides=(2,2), padding='same', kernel_initializer = conv_init)(x)
    x = normalize()(x, training=1)
    x = Activation('relu')(x)

    x = res_block(x, ngf*4)
    x = res_block(x, ngf*4)
    x = res_block(x, ngf*4)
    x = res_block(x, ngf*4)
    x = res_block(x, ngf*4)
    x = res_block(x, ngf*4)
    
    x = scaleup(x, ngf*2, (ks, ks), strides=(2,2), padding='same')
    x = normalize()(x, training=1)
    x = Activation('relu')(x)
    
    x = scaleup(x, ngf, (ks, ks), strides=(2,2), padding='same')
    x = normalize()(x, training=1)
    x = Activation('relu')(x)

    x = padding((p,p))(x)
    x = Conv2D(output_nc, (f,f), kernel_initializer = conv_init)(x)
    x = Activation('tanh')(x)

    model = Model(input, x, name=kwargs.get('name',None))
    print('Model resnet 6blocks:')
    model.summary()
    return model

def unet_128(input_shape, output_nc, ngf=64, fixed_input_size=True, **kwargs):    
    isize = input_shape[0]
    nc_in = input_shape[2]
    nc_out = output_nc
    max_nf = 8*ngf    
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'conv_{0}'.format(s)) (x)
        if s>2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s//2, nf_next)
            x = Concatenate(axis=3)([x, x2])            
        x = Activation("relu")(x)
        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init,          
                            name = 'convt.{0}'.format(s))(x)        
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        return x
    
    s = isize if fixed_input_size else None
    m = inputs = Input(shape=(s, s, nc_in))        
    m = block(m, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    m = Activation('tanh')(m)
    
    # Model
    model = Model(inputs=inputs, outputs=[m])
    print('Generator Unet 128:')
    model.summary()
    return model

