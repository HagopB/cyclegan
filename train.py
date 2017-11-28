import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path_trainA", type=str, help="The path to the A style images")
parser.add_argument("--path_trainB", type=str, help="The path to the B style images")
parser.add_argument("--pic_dir", type=str, help="Picture directory where to store all intermediate images")
parser.add_argument("--lmbd", type=int, help="Lambada - weight of cycleloss", default=10)
parser.add_argument("--lmbd_feat", type=int, help="Lambda - weight of perception loss", default=0)
parser.add_argument("--niter", type=int, help="Total number of iterations", default=200)
parser.add_argument("--save_iter", type=int, help="Number of iterations before saving the model", default=250)
parser.add_argument("--cuda", type=str, help="cuda", default='1')
args = parser.parse_args()
print(args)

import os
#os.environ['TENSORFLOW_FLAGS']=os.environ.get('TENSORFLOW_FLAGS','')+',gpuarray.preallocate=0.45,device=cuda{}'.format(args.cuda)
#os.environ['CUDA_VISIBLE_DEVICES']='{}'.format(args.cuda)

from utils import ImageGenerator
from models import CycleGAN
from utils import Option

if __name__ == '__main__':
    opt = Option()
    opt.batch_size = 1
    opt.save_iter = args.save_iter
    opt.niter = args.niter
    opt.lmbd = args.lmbd
    opt.pic_dir = args.pic_dir
    opt.idloss = 0.0
    opt.lr = 0.0002
    opt.d_iter = 1
    if args.lmbd_feat != 0:
        opt.perceptionloss = True
    else:
        opt.perceptionloss = False
    opt.lmbd_feat = args.lmbd_feat
    
    opt.__dict__.update(args.__dict__)
    opt.summary()


    cycleGAN = CycleGAN(opt)

    IG = ImageGenerator(path_trainA=args.path_trainA,
                        path_trainB=args.path_trainB,
                        resize=opt.resize,
                        crop=opt.crop)

    cycleGAN.fit(IG)
