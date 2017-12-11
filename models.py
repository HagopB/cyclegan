import os
from generator import defineG
from discriminator import defineD
import keras
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
from utils import ImageGenerator
import keras.backend as k
from layers import *
import numpy as np
import sys
from utils import  *
from scipy.misc import *
from resnet_builder import *

'''
*****************************************************************************
*********************************** Losses **********************************
*****************************************************************************
'''

def mse_loss(y_true, y_pred):
    #MSE
    loss = k.mean(k.square(y_true - y_pred))
    return loss

def cycle_loss(y_true, y_pred):
    # MAE
    loss = k.mean(k.abs(y_true - y_pred))
    return loss

def gan_loss(y_true, y_pred, use_lsgan=True):
    if use_lsgan == True:
        #MSE
        loss = k.mean(k.square(y_pred - y_true)) 
    else:
        loss = -k.mean(k.log(y_pred + 1e-12) * y_true + k.log(1 - y_pred + 1e-12) * (1 - y_true))
    return loss

def cycle_variables(netG1, netG2):
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    rec_input = netG2([fake_output])
    fn_generate = K.function([real_input], [fake_output, rec_input])
    return real_input, fake_output, rec_input, fn_generate

def feature_loss(netFeat, real_A, fake_B, rec_A, real_B, fake_A, rec_B):
    loss_AfB = mse_loss(netFeat([real_A]), netFeat([fake_B]))
    loss_BfA = mse_loss(netFeat([real_B]), netFeat([fake_A]))
    
    loss_fArecB = mse_loss(netFeat([fake_A]), netFeat([rec_B]))  
    loss_fBrecA = mse_loss(netFeat([fake_B]), netFeat([rec_A]))  
    
    loss_ArecA = mse_loss(netFeat([real_A]), netFeat([rec_A]))
    loss_BrecB = mse_loss(netFeat([real_B]), netFeat([rec_B]))
    
    return loss_AfB, loss_BfA, loss_fArecB, loss_fBrecA, loss_ArecA, loss_BrecB

def loss_(netD, real, fake, rec):
    output_real = netD([real])
    output_fake = netD([fake])
    
    # loss D
    loss_D_real = gan_loss(output_real, k.ones_like(output_real))
    loss_D_fake = gan_loss(output_fake, k.zeros_like(output_fake))
    loss_D = loss_D_real + loss_D_fake
    
    # loss G
    loss_G = gan_loss(output_fake, k.ones_like(output_fake))
    
    # loss cycle
    loss_cyc = cycle_loss(rec, real)
    return loss_D, loss_G, loss_cyc

'''
*****************************************************************************
*****************************   cycleGAN   **********************************
*****************************************************************************
'''
class BaseModel(object):
    name = 'BaseModel'
    def __init__(self):
        raise NotImplemented

    def save(self):
        raise NotImplemented

    def plot(self):
        raise NotImplemented

class CycleGAN(BaseModel):
    name = 'CycleGAN'
    def __init__(self, opt):
        netDA = defineD(opt.which_model_netD, input_shape=opt.shapeA, ndf=opt.ndf, use_sigmoid=False, name='netDA')
        netDB = defineD(opt.which_model_netD, input_shape=opt.shapeB, ndf=opt.ndf, use_sigmoid=False, name='netDB')

        netGA = defineG(opt.which_model_netG, input_shape=opt.shapeB, output_shape=opt.shapeA, ngf=opt.ngf, name='netGA')
        netGB = defineG(opt.which_model_netG, input_shape=opt.shapeA, output_shape=opt.shapeB, ngf=opt.ngf, name='netGB')       
        
        # generate variables
        real_A, fake_B, rec_A, cycleA_generate = cycle_variables(netGB, netGA)
        real_B, fake_A, rec_B, cycleB_generate = cycle_variables(netGA, netGB)
        
        # compute loss
        loss_DA, loss_GA, loss_cycA = loss_(netDA, real_A, fake_A, rec_A)
        loss_DB, loss_GB, loss_cycB = loss_(netDB, real_B, fake_B, rec_B)
        loss_cyc = loss_cycA + loss_cycB
        
        if opt.perceptionloss == True:
            netFeat = definenetFeat(input_shape=opt.shapeA, name='netFeat')
            # features loss
            loss_AfB, loss_BfA, loss_fArecB, loss_fBrecA, loss_ArecA, loss_BrecB = feature_loss(netFeat,
                                                                                                real_A, fake_B, rec_A,
                                                                                                real_B, fake_A, rec_B)
            
            loss_feat = opt.lmbd_feat * (loss_AfB + loss_BfA + loss_fArecB + loss_fBrecA + loss_ArecA + loss_BrecB)
            
            # Generator Loss: 
            loss_G = loss_GA + loss_GB + opt.lmbd * loss_cyc + loss_feat
            
            # build for Generator
            weightsG = netGA.trainable_weights + netGB.trainable_weights + netFeat.trainable_weights
       
            adam_g = Adam(lr=opt.lr, beta_1=0.5)
            training_updates_g = adam_g.get_updates(weightsG, [], loss_G)
        
            G_trainner = K.function([real_A, real_B],
                                    [loss_GA, loss_GB, loss_cyc, loss_feat],
                                    training_updates_g)
            
        else:    
        
            # Generator Loss: 
            loss_G = loss_GA + loss_GB + opt.lmbd * loss_cyc 
            
            # build for Generator
            weightsG = netGA.trainable_weights + netGB.trainable_weights
            
            adam_g = Adam(lr=opt.lr, beta_1=0.5)
            training_updates_g = adam_g.get_updates(weightsG, [], loss_G)
        
            G_trainner = K.function([real_A, real_B],
                                    [loss_GA, loss_GB, loss_cyc],
                                    training_updates_g)
        # Discriminator loss:
        loss_D = 0.5*(loss_DA + loss_DB)
        
        # build for Discriminator
        weightsD = netDA.trainable_weights + netDB.trainable_weights
        
        adam_d = Adam(lr=opt.lr, beta_1=0.5)
        training_updates_d = adam_d.get_updates(weightsD, [],loss_D)
        
        D_trainner = K.function([real_A, real_B],
                                [loss_DA/2, loss_DB/2],
                                training_updates_d)

        self.G_trainner = G_trainner
        self.D_trainner = D_trainner
        
        self.AtoB = netGB
        self.BtoA = netGA
        
        self.DisA = netDA
        self.DisB = netDB
        
        self.cycleA_generate = cycleA_generate
        self.cycleB_generate = cycleB_generate
        
        self.opt = opt
        self.adam_g = adam_g
        self.adam_d = adam_d
        
    def fit(self, img_generator):
        opt = self.opt
        # managing intermediate results directory
        if not os.path.exists(opt.pic_dir):
            os.mkdir(opt.pic_dir)
        
        # defining batch size
        bs = opt.batch_size


        train_batch = img_generator(bs)
        
        niter = opt.niter
        display_iters = 50
        epoch = 0
        iteration = 0
        errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = errFeat_sum = 0
        
        while epoch < opt.niter:
            epoch, A, B = next(train_batch)        
            
            # train discriminator
            errDA, errDB  = self.D_trainner([A, B])
            errDA_sum += errDA
            errDB_sum += errDB
            
            #train generator
            if opt.perceptionloss == True:
                errGA, errGB, errCyc, errFeat = self.G_trainner([A, B])
                errGA_sum += errGA
                errGB_sum += errGB
                errCyc_sum += errCyc
                errFeat_sum += errFeat
                
            else:
                errGA, errGB, errCyc = self.G_trainner([A, B])
                errGA_sum += errGA
                errGB_sum += errGB
                errCyc_sum += errCyc
              
            if iteration%50 == 0:
                if opt.perceptionloss == True:
                    to_print = '[{}/{}][{}] Loss_D: {} {} Loss_G: {} {} loss_cyc {} loss_feat {}'.format(epoch, niter, iteration,
                                                                                                     errDA_sum/50, errDB_sum/50,
                                                                                                     errGA_sum/50, errGB_sum/50,
                                                                                                     errCyc_sum/50, errFeat/50)
                else:
                     to_print = '[{}/{}][{}] Loss_D: {} {} Loss_G: {} {} loss_cyc {}'.format(epoch, niter, iteration,
                                                                                   errDA_sum/50, errDB_sum/50,
                                                                                   errGA_sum/50, errGB_sum/50,
                                                                                   errCyc_sum/50)     
                print(to_print)

            if iteration%opt.save_iter == 0:
                # save intermediate results
                _, A, B = train_batch.send(4)      
                
                assert A.shape==B.shape
                def G(fn_generate, X):
                    r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
                    return r.swapaxes(0,1)[:,:,0]        
                
                rA = G(self.cycleA_generate, A)
                rB = G(self.cycleB_generate, B)
                arr = np.concatenate([A,B,rA[0],rB[0],rA[1],rB[1]])
                saveX(arr, os.path.join(opt.pic_dir, 'int_res.png'), 3)
                
                errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = errFeat_sum = 0 
            
            if iteration%2500 == 0:
                # save model
                self.AtoB.save(os.path.join(opt.pic_dir, 'a2b.h5'))
                self.BtoA.save(os.path.join(opt.pic_dir, 'b2a.h5'))
                
                
            iteration += bs
            
    def predict(self, path_images, model_path):
        opt = self.opt
        if not os.path.exists(opt.pic_dir):
            os.mkdir(opt.pic_dir)
            
        print('Predicting with model' + model_path.split('.')[0])
        model = keras.models.load_model(model_path, 
                                        custom_objects={'InstanceNormalization2D': InstanceNormalization2D})
        
        print('Predicting for all images')
        
        test_data_path = path_images
        images = os.listdir(test_data_path)
        total = len(images)

        imgs = np.ndarray((total, opt.shapeA[0], opt.shapeA[1], opt.shapeA[2]), dtype=np.uint8)
        imgs_id = np.ndarray((total, ), dtype=np.int32)
        
        print('Creating test images')
        for idx, image_name in enumerate(images):
            img_id = int(image_name.split('.')[0])

            img = imread(os.path.join(test_data_path, image_name))
            img =  imresize(img, opt.crop)

            imgs[idx] = img
            imgs_id[idx] = img_id
            if idx % 100 == 0:
                print('Done: {0}/{1} images'.format(idx, total))

        print('Predicting test images')
        imgs = imgs/127.5-1
        preds = model.predict(imgs,batch_size=1,verbose=1)

        for idx, e in enumerate(preds):
            imsave(os.path.join(opt.pic_dir,'{}.png'.format(imgs_id[idx])),e)
            if idx % 100 == 0:
                print('Done: {0}/{1} images'.format(idx, total))
        
        print('Preparing demo image')
        real = imgs[:20]             
        img = vis_grid(np.concatenate([real[:5], preds[:5],real[5:10],preds[5:10],
                                      real[10:15],preds[10:15],real[15:],preds[15:]],axis=0),
                       8, 5,os.path.join(opt.pic_dir, 'demo.png'))       
        