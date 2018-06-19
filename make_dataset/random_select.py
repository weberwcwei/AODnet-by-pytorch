import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', type=str, required=True, help='path to train directory')
parser.add_argument('--valroot', type=str, required=True, help='path to validation directory')
args = parser.parse_args()
print(args)

train = args.trainroot
val = args.valroot

if not os.path.exists(val):
    os.makedirs(val)
for i in range(3169):
    a = random.choice(os.listdir(train))
    shutil.move((train+'/'+a), (val+'/'+a))
