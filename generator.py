from keras.layers import UpSampling2D
from keras.layers import add
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
    #return batchnorm()#axis=get_filter_dim()
    return InstanceNormalization2D()


def resnet_block(input, dim, ks =(3,3), strides=(1,1)):
    x = padding((1,1))(input)
    x = Conv2D(dim, ks,strides=strides, kernel_initializer=conv_init)(x)
    x = normalize()(x)
    x = Activation('relu')(x)

    x = padding((1,1))(x)
    x = Conv2D(dim, ks,strides=strides, kernel_initializer=conv_init)(x)
    x = normalize()(x)
    res = add([input, x])
    return res


def resnet_6blocks(input_shape, output_nc, ngf, **kwargs):
    input = Input(input_shape)
    x = padding((3,3))(input)
    x = Conv2D(ngf, (7,7), kernel_initializer=conv_init)(x)
    x = normalize()(x)
    x = Activation('relu')(x)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(ngf * mult * 2, (3,3),
                   padding='same', strides=(2,2),
                   kernel_initializer=conv_init)(x)
        x = normalize()(x)
        x = Activation('relu')(x)


    mult = 2**n_downsampling
    for i in range(6):
        x = resnet_block(x, ngf * mult)

    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        f = int(ngf * mult / 2)
        x = Conv2DTranspose(f, (3,3), strides=(2,2),
                            padding='same', kernel_initializer=conv_init)(x)
        x = normalize()(x)
        x = Activation('relu')(x)

    x = padding((3,3))(x)
    x = Conv2D(output_nc, (7,7), kernel_initializer = conv_init)(x)
    x = Activation('tanh')(x)


    model = Model(inputs=input, outputs=[x])
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

