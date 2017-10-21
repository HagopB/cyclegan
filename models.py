import os
from generator import defineG
from discriminator import defineD
import keras
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
from utils import ImageGenerator
import keras.backend as k
from layers import InstanceNormalization2D
import numpy as np
import sys
from utils import  vis_grid
from scipy.misc import *


'''
*****************************************************************************
*****************************   cycle loss **********************************
*****************************************************************************
'''
def cycle_loss(y_true, y_pred):
    if k.image_data_format() is 'channels_first':
        x_w = 2
        x_h = 3
    else:
        x_w = 1
        x_h = 2
    loss = k.abs(y_true - y_pred)
    loss = k.sum(loss, axis=x_h)
    loss = k.sum(loss, axis=x_w)
    return loss
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
    @staticmethod
    def init_network(model):
        for w in model.weights:
            if w.name.startswith('conv2d') and w.name.endswith('kernel'):
                value = np.random.normal(loc=0.0, scale=0.02, size=w.get_value().shape)
                w.set_value(value.astype('float32'))
            if w.name.startswith('conv2d') and w.name.endswith('bias'):
                value = np.zeros(w.get_value().shape)
                w.set_value(value.astype('float32'))

    def __init__(self, opt):
        gen_B = defineG(opt.which_model_netG, input_shape=opt.shapeA, output_shape=opt.shapeB, ngf=opt.ngf, name='gen_B')
        dis_B = defineD(opt.which_model_netD, input_shape=opt.shapeB, ndf=opt.ndf, use_sigmoid=not opt.use_lsgan, name='dis_B')

        gen_A = defineG(opt.which_model_netG, input_shape=opt.shapeB, output_shape=opt.shapeA, ngf=opt.ngf, name='gen_A')
        dis_A = defineD(opt.which_model_netD, input_shape=opt.shapeA, ndf=opt.ndf, use_sigmoid=not opt.use_lsgan, name='dis_A')

        self.init_network(gen_B)
        self.init_network(dis_B)
        self.init_network(gen_A)
        self.init_network(dis_A)


        # build for generators
        real_A = Input(opt.shapeA)
        fake_B = gen_B(real_A)
        dis_fake_B = dis_B(fake_B)
        rec_A = gen_A(fake_B) # = gen_A(gen_B(real_A))

        real_B = Input(opt.shapeB)
        fake_A = gen_A(real_B)
        dis_fake_A = dis_A(fake_A)
        rec_B = gen_B(fake_A) # = gen_B(gen_A(real_B))

        if opt.idloss > 0:
            G_trainner = Model([real_A, real_B],
                     [dis_fake_B,   dis_fake_A,     rec_A,      rec_B,      fake_B,     fake_A])

            G_trainner.compile(Adam(lr=opt.lr, beta_1=opt.beta1,),
                loss=['MSE',        'MSE',          cycle_loss,      cycle_loss,      cycle_loss,      cycle_loss],
                loss_weights=[1,    1,              opt.lmbd,   opt.lmbd,   opt.idloss  ,opt.idloss])
        else:
            G_trainner = Model([real_A, real_B],
                     [dis_fake_B,   dis_fake_A,     rec_A,      rec_B,      ])

            G_trainner.compile(Adam(lr=opt.lr, beta_1=opt.beta1,),
                loss=['MSE',        'MSE',          cycle_loss,      cycle_loss,      ],
                loss_weights=[1,    1,              opt.lmbd,   opt.lmbd,   ])
        # label:  0             0               real_A      real_B


        # build for discriminators
        real_A = Input(opt.shapeA)
        fake_A = Input(opt.shapeA)
        real_B = Input(opt.shapeB)
        fake_B = Input(opt.shapeB)

        dis_real_A = dis_A(real_A)
        dis_fake_A = dis_A(fake_A)
        dis_real_B = dis_B(real_B)
        dis_fake_B = dis_B(fake_B)

        D_trainner = Model([real_A, fake_A, real_B, fake_B],
                [dis_real_A, dis_fake_A, dis_real_B, dis_fake_B])
        D_trainner.compile(Adam(lr=opt.lr, beta_1=opt.beta1,), loss='MSE')
        # label: 0           0.9         0           0.9


        self.G_trainner = G_trainner
        self.D_trainner = D_trainner
        self.AtoB = gen_B
        self.BtoA = gen_A
        self.DisA = dis_A
        self.DisB = dis_B
        self.opt = opt

    def fit(self, img_A_generator, img_B_generator):
        opt = self.opt
        if not os.path.exists(opt.pic_dir):
            os.mkdir(opt.pic_dir)
        bs = opt.batch_size

        fake_A_pool = []
        fake_B_pool = []

        iteration = 0
        while iteration < opt.niter:
            print('iteration: {}'.format(iteration))
            # sample
            real_A = img_A_generator(bs)
            real_B = img_B_generator(bs)

            # fake pool
            fake_A_pool.extend(self.BtoA.predict(real_B))
            fake_B_pool.extend(self.AtoB.predict(real_A))
            fake_A_pool = fake_A_pool[-opt.pool_size:]
            fake_B_pool = fake_B_pool[-opt.pool_size:]

            fake_A = [fake_A_pool[ind] for ind in np.random.choice(len(fake_A_pool), size=(bs,), replace=False)]
            fake_B = [fake_B_pool[ind] for ind in np.random.choice(len(fake_B_pool), size=(bs,), replace=False)]
            fake_A = np.array(fake_A)
            fake_B = np.array(fake_B)

            ones  = np.ones((bs,)+self.G_trainner.output_shape[0][1:])
            zeros = np.zeros((bs, )+self.G_trainner.output_shape[0][1:])


            # train
            for _ in range(opt.d_iter):
                _, D_loss_real_A, D_loss_fake_A, D_loss_real_B, D_loss_fake_B = \
                    self.D_trainner.train_on_batch([real_A, fake_A, real_B, fake_B],
                        [zeros, ones*0.9, zeros, ones*0.9])


            if opt.idloss > 0:
                _, G_loss_fake_B, G_loss_fake_A, G_loss_rec_A, G_loss_rec_B, G_loss_id_A, G_loss_id_B = \
                    self.G_trainner.train_on_batch([real_A, real_B],
                        [zeros, zeros, real_A, real_B, real_A, real_B])
            else:
                _, G_loss_fake_B, G_loss_fake_A, G_loss_rec_A, G_loss_rec_B = \
                    self.G_trainner.train_on_batch([real_A, real_B],
                        [zeros, zeros, real_A, real_B, ])


            print('Generator Loss:')
            print('fake_B: {} rec_A: {} | fake_A: {} rec_B: {}'.\
                    format(G_loss_fake_B, G_loss_rec_A, G_loss_fake_A, G_loss_rec_B))
            if opt.idloss > 0:
                print('id_loss_A: {}, id_loss_B: {}'.format(G_loss_id_A, G_loss_id_B))

            print('Discriminator Loss:')
            print('real_A: {} fake_A: {} | real_B: {} fake_B: {}'.\
                    format(D_loss_real_A, D_loss_fake_A, D_loss_real_B, D_loss_fake_B))


            print("Dis_A")
            res = self.DisA.predict(real_A)
            print("real_A: {}".format(res.mean()))
            res = self.DisA.predict(fake_A)
            print("fake_A: {}".format(res.mean()))

            if iteration % opt.save_iter == 0:
                imga = real_A
                imga2b = self.AtoB.predict(imga)
                imga2b2a = self.BtoA.predict(imga2b)

                imgb = real_B
                imgb2a = self.BtoA.predict(imgb)
                imgb2a2b = self.AtoB.predict(imgb2a)

                vis_grid(np.concatenate([imga, imga2b, imga2b2a, imgb, imgb2a, imgb2a2b],
                                                                            axis=0),
                        6, bs, os.path.join(opt.pic_dir, '{}.png'.format(iteration)) )

                self.AtoB.save(os.path.join(opt.pic_dir, 'a2b.h5'))
                self.BtoA.save(os.path.join(opt.pic_dir, 'b2a.h5'))

#               import ipdb
#               ipdb.set_trace()
            iteration += 1
            sys.stdout.flush()
    def predict(self, path_images, model_path):
        opt = self.opt
 
        if not os.path.exists(opt.pic_dir):
            os.mkdir(opt.pic_dir)
			
        img_generator = ImageGenerator(path_images,resize=opt.resize,crop=opt.crop)
        print('Predicting with model' + model_path.split('.')[0])
        
        print('Preparing demo image')
        real = img_generator(20)
        
        model = keras.models.load_model(model_path,custom_objects={'InstanceNormalization2D': InstanceNormalization2D})
        preds = model.predict(real)                
        img = vis_grid(np.concatenate([real[:5], preds[:5],real[5:10],preds[5:10],
                                      real[10:15],preds[10:15],real[15:],preds[15:]],axis=0),
                       8, 5,os.path.join(opt.pic_dir, 'demo.png'))       
        
        
        print('Predicting for all images')
        
        test_data_path = path_images
        images = os.listdir(test_data_path)
        total = len(images)

        imgs = np.ndarray((total, 256, 256, 3), dtype=np.uint8)
        imgs_id = np.ndarray((total, ), dtype=np.int32)
        
        print('Creating test images')
        for idx, image_name in enumerate(images):
            img_id = int(image_name.split('.')[0])

            img = imread(os.path.join(test_data_path, image_name))
            img =  imresize(img,(256, 256))

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
        
