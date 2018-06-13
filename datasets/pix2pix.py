import glob
import os
import os.path

import numpy as np
import scipy.ndimage
import torch.utils.data as data
from PIL import Image

import h5py

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
  images = []
  if not os.path.isdir(dir):
    raise Exception('Check dataroot')
  for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
      path = os.path.join(dir, fname)
      item = path
      images.append(item)
  return images

def default_loader(path):
  return Image.open(path).convert('RGB')

class pix2pix(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    imgs = make_dataset(root)
    self.root = root
    self.imgs = imgs
    self.transform = transform
    self.loader = loader

    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):
    path = self.imgs[index]
    f=h5py.File(path,'r')

    haze_image=f['haze'][:]
    GT=f['gt'][:]

    haze_image=np.swapaxes(haze_image,0,2)
    GT=np.swapaxes(GT,0,2)

    haze_image=np.swapaxes(haze_image,1,2)
    GT=np.swapaxes(GT,1,2)
    return haze_image, GT

  def __len__(self):
    train_list=glob.glob(self.root+'/*h5')
    return len(train_list)
