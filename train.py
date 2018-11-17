import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import AODnet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='pix2pix',  help='')
parser.add_argument('--dataroot', required=True, help='path to trn dataset')
parser.add_argument('--valDataroot', required=True, help='path to val dataset')
parser.add_argument('--valBatchSize', type=int, default=32, help='input batch size')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use, if Your OS is window, please set to 0')
parser.add_argument('--exp', default='pretrain', help='folder to model checkpoints')
parser.add_argument('--printEvery', type=int, default=50, help='number of batches to print average loss ')
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--epochSize', type=int, default=840, help='number of batches as one epoch (for validating once)')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs for training')

args = parser.parse_args()
print(args)

args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
print("Random Seed: ", args.manualSeed)

#===== Dataset =====
def getLoader(datasetName, dataroot, batchSize, workers,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True, seed=None):

  if datasetName == 'pix2pix':
    from datasets.pix2pix import pix2pix as commonDataset
    import transforms.pix2pix as transforms
  if split == 'train':
    dataset = commonDataset(root=dataroot,
                            transform=transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                            ]),
                            seed=seed)
  else:
    dataset = commonDataset(root=dataroot,
                            transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                             ]),
                             seed=seed)

  dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batchSize, 
                                           shuffle=shuffle, 
                                           num_workers=int(workers))
  return dataloader

trainDataloader = getLoader(args.dataset,
                       args.dataroot,
                       args.batchSize,
                       args.threads,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=args.manualSeed)

valDataloader = getLoader(args.dataset,
                          args.valDataroot,
                          args.valBatchSize,
                          args.threads,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=args.manualSeed)

#===== DehazeNet =====
print('===> Building model')
net = AODnet()
if args.cuda:
    net = net.cuda()

#===== Loss function & optimizer =====
criterion = torch.nn.MSELoss()

if args.cuda:
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=53760, gamma=0.5)

#===== Training and validation procedures =====

def train(epoch):
    net.train()
    epoch_loss = 0
    for iteration, batch in enumerate(trainDataloader, 0):
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        varIn, varTar = varIn.float(), varTar.float()

        if args.cuda:
            varIn = varIn.cuda()
        if args.cuda:
            varTar = varTar.cuda()
            
        # print(iteration)
        optimizer.zero_grad()

        loss = criterion(net(varIn), varTar)
        # print(loss)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if iteration%args.printEvery == 0:
            print("===> Epoch[{}]({}/{}): Avg. Loss: {:.4f}".format(epoch, iteration+1, len(trainDataloader), epoch_loss/args.printEvery))
            epoch_loss = 0

def validate():
    net.eval()
    avg_mse = 0
    for _, batch in enumerate(valDataloader, 0):
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        varIn, varTar = varTar.float(), varIn.float()
        
        if args.cuda:
            varIn = varIn.cuda()
        if args.cuda:
            varTar = varTar.cuda()

        prediction = net(varIn)
        mse = criterion(prediction, varTar)
        avg_mse += mse.data[0]
    print("===>Avg. Loss: {:.4f}".format(avg_mse/len(valDataloader)))

 
def checkpoint(epoch):
    model_out_path = "./model_pretrained/AOD_net_epoch_relu_{}.pth".format(epoch)
    torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
#===== Main procedure =====
for epoch in range(1, args.nEpochs + 1):
    train(epoch)
    validate()
    checkpoint(epoch)
