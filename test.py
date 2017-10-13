import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path_test", type=str, help="The path to the test images")
parser.add_argument("--pic_dir", type=str, help="Picture directory where to store all predicte images")
parser.add_argument("--model_path",type=str,help="Path to the keras model .h5 or .hdf5")
parser.add_argument("--cuda", type=str, help="cuda", default='2')
args = parser.parse_args()
print(args)

import os
from utils import *
from models import *
from utils import *
from models import *
from utils import *
from layers import *

if __name__ == '__main__':
    opt = Option()
    opt.pic_dir = args.pic_dir
    
    opt.__dict__.update(args.__dict__)
    opt.summary()

    cycleGAN = CycleGAN(opt)
    
    cycleGAN.predict(args.path_test, args.model_path)