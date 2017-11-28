from layers import *

'''
*****************************************************************************
********************************  defineD   *********************************
*****************************************************************************
'''
def defineD(which_model_netD, input_shape, ndf, use_sigmoid=False, **kwargs):
    if which_model_netD == 'basic':
        return basic_D(input_shape, ndf, use_sigmoid=use_sigmoid, **kwargs)
    else:
        raise NotImplemented

'''
*****************************************************************************
****************************  Discriminator  ********************************
*****************************************************************************
'''
def basic_D(input_shape, ndf, max_layers=3, use_sigmoid=False, **kwargs):
    
    nc_in  = input_shape[2]
    input_d = Input(input_shape)

    x = conv2d(ndf, kernel_size=4, strides=2, padding="same", name = 'First')(input_d)
    x = LeakyReLU(alpha=0.2)(x)
    
    for layer in range(1, max_layers):        
        out_feat = ndf * min(2**layer, 8)
        x = conv2d(out_feat, kernel_size=4, strides=2, padding="same", 
                   use_bias=False, name = 'pyramid.{0}'.format(layer))(x)
        x = batchnorm()(x, training=1)        
        x = LeakyReLU(alpha=0.2)(x)
    
    out_feat = ndf*min(2**max_layers, 8)
    x = ZeroPadding2D(1)(x)
    x = conv2d(out_feat, kernel_size=4,  use_bias=False, name = 'pyramid_last')(x)
    x = batchnorm()(x, training=1)
    x = LeakyReLU(alpha=0.2)(x)
    
    # final layer
    x = ZeroPadding2D(1)(x)
    if use_sigmoid == True:
        activation = "sigmoid"
    else:
        activation = None
    x = conv2d(1, kernel_size=4, name = 'final'.format(out_feat, 1), activation = activation)(x)
    
    # Model
    model = Model(inputs=[input_d], outputs = x)
    print('Model Basic Discriminator:')
    model.summary()
    return model

'''
def basic_D(input_shape, ndf, n_layers=3, kw=4, dropout=0.0, use_sigmoid=False, **kwargs):
    padw = (kw-1)/2

    input = Input(input_shape)
    x = Conv2D(ndf, (kw,kw), strides=(2,2), padding='same')(input)
    x = LeakyReLU(0.2)(x)

    nf_mult = 1
    for n in range(1, n_layers):
        nf_mult_prev = nf_mult
        nf_mult = min(2**n, 8)
        
        x = Conv2D(ndf*min(2**n, 8), (kw,kw), strides=(2,2), padding='same')(x)
        x = normalize()(x)
        if dropout > 0.: x = Dropout(dropout)(x)
        x = LeakyReLU(0.2)(x)
    
    
    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers, 8)
    
    x = Conv2D(ndf*nf_mult, (kw,kw), strides=(1,1), padding='same')(x)
    x = normalize()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1, (kw,kw), strides=(1,1), padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    model = Model(input, x, name=kwargs.get('name',None))
    print('Model basic D:')
    model.summary()

    return model
'''