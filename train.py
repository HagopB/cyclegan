import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path_trainA", type=str, help="The path to the A style images")
parser.add_argument("--path_trainB", type=str, help="The path to the B style images")
parser.add_argument("--pic_dir", type=str, help="picture directory where to store all intermediate images")
parser.add_argument("--niter", type=int, help="Total number of iterations", default=100000)
parser.add_argument("--save_iter", type=int, help="Number of iterations before saving the model", default=250)
parser.add_argument("--cuda", type=str, help="cuda", default='2')
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
    opt.lmbd = 10
    opt.pic_dir = args.pic_dir
    opt.idloss = 0.0
    opt.lr = 0.0001
    opt.d_iter = 1

    opt.__dict__.update(args.__dict__)
    opt.summary()


    cycleGAN = CycleGAN(opt)

    IG_A = ImageGenerator(root=args.path_trainA,
                resize=opt.resize,
                crop=opt.crop)
    IG_B = ImageGenerator(root=args.path_trainB,
                resize=opt.resize,
                crop=opt.crop)

    cycleGAN.fit(IG_A, IG_B)
