import argparse

import PIL.Image as Image
import scipy.misc
import skimage.io as sio
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, required=True, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda?')

args = parser.parse_args()
print(args)

#===== load model =====
print('===> Loading model')
net = torch.load(args.model)
if args.cuda:
    net = net.cuda()

#===== Load input image =====
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]
)

img = Image.open(args.input_image).convert('RGB')
imgIn = transform(img).unsqueeze_(0)

#===== Test procedures =====
varIn = Variable(imgIn)
if args.cuda:
    varIn = varIn.cuda()

prediction = net(varIn)
prediction = prediction.data.cpu().numpy().squeeze().transpose((1,2,0))
scipy.misc.toimage(prediction).save(args.output_filename)
