import keras.backend as K
from scipy.misc import *
import numpy as np
from pprint import pprint
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image
import numpy as np
import glob
from random import randint, shuffle

'''
*****************************************************************************
**************************   VISUAL UTILS    ********************************
*****************************************************************************
'''
def vis_grid(X, nh, nw, save_path=None):
    if X.shape[1] in [1,3,4]:
        X = X.transpose(0, 2, 3, 1)

    h, w = X.shape[1:3]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = int(n%nw)
        if n >= nh*nw: break
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    if save_path is not None:
        imsave(save_path, img)
    return img

def showG(A, B, path):
    assert A.shape==B.shape
    def G(fn_generate, X):
        r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
        return r.swapaxes(0,1)[:,:,0]        
    rA = G(cycleA_generate, A)
    rB = G(cycleB_generate, B)
    arr = np.concatenate([A,B,rA[0],rB[0],rA[1],rB[1]])
    saveX(arr, 3, path)
    
def saveX(X, path, rows=1):
    imageSize=128
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')
    int_X = int_X.reshape(-1,imageSize,imageSize, 3)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
    img = Image.fromarray(int_X)
    img.save(path)

    

'''
*****************************************************************************
**************************   ImageGenerator  ********************************
*****************************************************************************
'''

class ImageGenerator(object):
    '''ImageGenerator'''
    def __init__(self,
                 path_trainA,
                 path_trainB,
                 resize=None,
                 crop=None):
        self.n_images_trainA_ = len(os.listdir(path_trainA))
        self.n_images_trainB_ = len(os.listdir(path_trainB))
        self.path_trainA = path_trainA
        self.path_trainB = path_trainB
        self.resize = resize
        self.crop = crop
        
    def read_image(self, fn):
        im = Image.open(fn).convert('RGB')
        im = im.resize(self.resize, Image.BILINEAR )
        arr = np.array(im)/255*2-1
        w1, w2 = (self.resize[0] - self.crop[0])//2, (self.resize[0] + self.crop[0])//2
        h1, h2 = w1,w2
        img = arr[h1:h2, w1:w2, :]
        if randint(0,1):
            img=img[:,::-1]
        return img

    def minibatch(self, data, bs):
        length = len(data)
        epoch = i = 0
        tmpsize = None    
        while True:
            size = tmpsize if tmpsize else bs
            if i + size > length:
                shuffle(data)
                i = 0
                epoch+=1        
            rtn = [self.read_image(data[j]) for j in range(i, i + size)]
            i += size
            tmpsize = yield epoch, np.float32(rtn)
    
    def minibatchAB(self, dataA, dataB, bs):
        batchA = self.minibatch(dataA, bs)
        batchB = self.minibatch(dataB, bs)
        tmpsize = None    
        while True:        
            ep1, A = batchA.send(tmpsize)
            ep2, B = batchB.send(tmpsize)
            tmpsize = yield max(ep1, ep2), A, B
            
    def __call__(self, bs):
        trainA = glob.glob('{}/*'.format(self.path_trainA))
        trainB = glob.glob('{}/*'.format(self.path_trainB))

        print('N images train A {} -- N images train B {}'.format(len(trainA), len(trainB)))

        return self.minibatchAB(trainA, trainB, bs)
               

'''
*****************************************************************************
*******************************   OPTIONS   *********************************
*****************************************************************************
'''
class Option(object):
    '''
        Option parameters
    '''
    def __init__(self,
        # from CycleGAN/options.lua
        # data
        DATA_ROOT = '',                     # path to images (should have subfolders 'train', 'val', etc)
        shapeA = (128,128,3),               #(256,256,3),
        shapeB = (128,128,3),               #(256,256,3),
        resize = (143,143),                 #(286,286),
        crop   = (128,128),                 #(256,256),

        # net definition
        which_model_netD = 'basic',         # selects model to use for netD
        which_model_netG = 'resnet_6blocks',      # selects model to use for netG
        use_lsgan = 1,                      # if 1, use least square GAN, if 0, use vanilla GAN
        perceptionloss = False,    # wether to use CycleGan with perception loss
        ngf = 64,                           # #  of gen filters in first conv layer
        ndf = 64,                           # of discrim filters in first conv layer
        lmbd = 10.0,
        lmbd_feat = 1.0,                    
                 
        # optimizers
        lr = 0.0002,                        # initial learning rate for adam
        beta1 = 0.5,                        # momentum term of adam

        # training parameters
        batch_size = 1,                     # images in batch
        niter = 100,                        #  of iter at starting learning rate
        pool_size = 50,                     # the size of image buffer that stores previously generated images
        save_iter = 50,
        d_iter = 10,

        # dirs
        pic_dir = 'quickshots',

        niter_decay = 100,                  # of iter to linearly decay learning rate to zero
        ntrain = np.inf,                    #  of examples per epoch. math.huge for full dataset
        flip = 1,                           # if flip the images for data argumentation
        display_id = 10,                    # display window id.
        display_winsize = 128,              # 256 if images are of shape (256, 256, 3)
        display_freq = 25,                  # display the current results every display_freq iterations
        gpu = 1,                            # gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
        cuda=1,
        name = '',                          # name of the experiment, should generally be passed on the command line
        save_epoch_freq = 1,                # save a model every save_epoch_freq epochs (does not overwrite previously saved models)
        save_latest_freq = 5000,            # save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
        print_freq = 50,                    # print the debug information every print_freq iterations
        save_display_freq = 2500,           # save the current display of results every save_display_freq_iterations
    ):
        #assert shapeA[0:1] == crop
        self.__dict__.update(locals())

    def summary(self):
        pprint(self.__dict__)
