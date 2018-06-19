from __future__ import division

# torch condiguration
import argparse
import math
import os
import pdb
import pickle
import random
import shutil
import sys
import time
from math import log10
from random import uniform

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
# import scipy.io as sio
import numpy as np
import PIL
import scipy
import scipy.io as sio
import scipy.ndimage.interpolation
from PIL import Image

import h5py

sys.path.append("./mingqingscript")
plt.ion()

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255 * np.ones((len(arr), 1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

parser = argparse.ArgumentParser()
parser.add_argument('--nyu', type=str, required=True, help='path to nyu_depth_v2_labeled.mat')
parser.add_argument('--dataset', type=str, require=True, help='path to  synthesized hazy images dataset store')
args = parser.parse_args()
print(args)

index = 1
nyu_depth = h5py.File(args.nyu + '/nyu_depth_v2_labeled.mat', 'r')

directory = args.dataset + '/train'
saveimgdir = args.dataset + '/demo'
if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(saveimgdir):
    os.makedirs(saveimgdir)

image = nyu_depth['images']
depth = nyu_depth['depths']

img_size = 224

# per=np.random.permutation(1400)
# np.save('rand_per.py',per)
# pdb.set_trace()
total_num = 0
plt.ion()
for index in range(1445):
    index = index
    gt_image = (image[index, :, :, :]).astype(float)
    gt_image = np.swapaxes(gt_image, 0, 2)
    gt_image = scipy.misc.imresize(gt_image, [480, 640]).astype(float)
    gt_image = gt_image / 255

    gt_depth = depth[index, :, :]
    maxhazy = gt_depth.max()
    minhazy = gt_depth.min()
    gt_depth = (gt_depth) / (maxhazy)

    gt_depth = np.swapaxes(gt_depth, 0, 1)

    for j in range(7):
        for k in range(3):
            #beta
            bias = 0.05
            temp_beta = 0.4 + 0.2*j
            beta = uniform(temp_beta-bias, temp_beta+bias)

            tx1 = np.exp(-beta * gt_depth)
            
            #A
            abias = 0.1
            temp_a = 0.5 + 0.2*k
            a = uniform(temp_a-abias, temp_a+abias)
            A = [a,a,a]

            m = gt_image.shape[0]
            n = gt_image.shape[1]

            rep_atmosphere = np.tile(np.reshape(A, [1, 1, 3]), [m, n, 1])
            tx1 = np.reshape(tx1, [m, n, 1])

            max_transmission = np.tile(tx1, [1, 1, 3])

            haze_image = gt_image * max_transmission + rep_atmosphere * (1 - max_transmission)

            total_num = total_num + 1
            scipy.misc.imsave(saveimgdir+'/haze.jpg', haze_image)
            scipy.misc.imsave(saveimgdir+'/gt.jpg', gt_image)

            h5f=h5py.File(directory+'/'+str(total_num)+'.h5','w')
            h5f.create_dataset('haze',data=haze_image)
            h5f.create_dataset('gt',data=gt_image)
