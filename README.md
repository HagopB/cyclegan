# CycleGAN #

### What is this repository for? 
Implementation of CycleGan model in Keras, ([original implementation link](https://github.com/junyanz/CycleGAN))

### How do I get set up ?  
Install Anaconda 3
Import the conda environment named `deepenv` using : 
```
conda env create -f deepenv.yml
```

Activate that environment using :
```
source activate deepenv
```
Now all the dependencies must be installed without problems (Keras 2, tensorflow 1 ...)

### How do I train CycleGAN with new images ?
you may have information on how to run ```train.py``` by:
```
python predict.py --help
```
you can train your own model by running (N.B.: example):
```
python train.py --path_trainA ./data/trainA --path_trainB ./data/trainB --pic_dir ./intermediate_res
```
### How do I deploy CycleGAN on new images after training?
you can deploy the model on a given collection, in order to transform A to B or B to A (Possible only after training).
```
python test.py --path_images ./data/trainA --pic_dir ./results --model_path ./../a2b.h5
```
### Contents
```
└── cyclegan
    ├── data                         # data folder contaning both A and B images
         ├── trainA                  # images belonging to class A
         └── trainB                  # images belonging to class B
    ├── pics                         # intermediate results folders (for training phase)
    ├── deepenv.yml                  # Environment (keras 2, tensorflow 1.1, etc ...)
    ├── discriminator.py             # discriminator
    ├── generator.py                 # generator (Resblock 6)
    ├── layers.py                    # ReflectPadding2D & InstanceNormalization2D
    ├── models.py                    # cycleGAN: fit & predict
    ├── README.md                    # Readme
    ├── test.py                      # deploy model
    ├── train.py                     # train model
    ├── utils.py                     # utils

```
### Demonstration: De-raining images 
The example below present 20 rainy images where cycleGAN has been used to de-rain.

![](https://github.com/HagopB/cyclegan/blob/master/pics/demo.png)


