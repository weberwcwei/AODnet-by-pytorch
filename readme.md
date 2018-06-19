# AOD-Net by Pytorch
This is an implementation of [AOD-Net : All-in-One Network for Dehazing](https://arxiv.org/abs/1707.06543) on Python3, Pytorch. The model can removal hazy, smoke or even water impurities.

The repository includes:
* Source code of AOD-Net
* Building code for synthesized hazy images based on [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
* Training code for our hazy dataset
* Pre-trained model for AOD-Net

# Requirements
Python 3.6, Pytorch 0.4.0 and other common packages

## NYU Depth V2
To build synthetic hazy dataset, you'll also need:
* Download [NYU Depth V2 labeled dataset](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)

# Training Part
## Dateset Setup
1. Clone this repository
2. Create dataset from the repository root directory
    ```bash
    $ cd make_dataset
    $ python create_train.py --nyu {Your NYU Depth V2 path} --dataset {Your trainset path}
    ``` 
3. Random pick 3,169 pictures as validation set
    ```bash
    $ python random_select.py --traindir {Your trainset path} --valdir {Your valset path}
    ```
## Start to training
4. training AOD-Net
    ```bash
    $ python train.py --dataroot {Your trainset path} --valDataroot {Your valset path} --cuda
    ```
# Testing Part
5. test hazy image on AOD-Net
    ```bash
    $ python test.py --input_image /test/canyon1.jpg  --model /model_pretrained/AOD_net_epoch_relu_10.pth --output_filename /result/canyon1_dehaze.jpg --cuda
    ```
