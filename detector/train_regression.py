from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from data import CTDataset
from regression import PointNetReg
import torch.nn.functional as F
from focal_loss import FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='pth', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')

parser.add_argument('--threshold_min', type=int, default=1700, help='minimum intensity threshold')
parser.add_argument('--threshold_max', type=int, default=2700, help='maximum intensity threshold')
parser.add_argument('--npoints', type=int, default=50000, help='number of points to sample')
parser.add_argument('--gamma', type=int, default=2, help='gamma for focal loss')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = CTDataset(root='../data',
                    threshold_min=int(opt.threshold_min),
                    threshold_max=int(opt.threshold_max),
                    npoints=opt.npoints, dim4=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

test_dataset = CTDataset(root='../data',
                         threshold_min=int(opt.threshold_min),
                         threshold_max=int(opt.threshold_max),
                         npoints=opt.npoints,
                         train=False, dim4=True)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

print("# of training examples: {0}".format(len(dataset)))
print("# of  testing examples: {0}".format(len(test_dataset)))
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetReg(num_points=opt.npoints)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.SGD(classifier.parameters(), lr=0.0001, momentum=0.9)
classifier.cuda()

focal_loss = FocalLoss(opt.gamma)
focal_loss = focal_loss.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, _, target, _ = data
        points, target = Variable(points), Variable(target)
        points = points.transpose(2, 1)

        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        pred, _ = classifier(points)
        #pred = pred.view(-1, 3)
        #target = target.view(-1, 3)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        print("[{0}: {1}/{2}] train loss: {3}".format(
            epoch, i, num_batch,
            loss.item(),
        ))
        if i % 32 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, _, target, _ = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            pred, _ = classifier(points)
            #pred = pred.view(-1, 3)
            #target = target.view(-1, 3)

            loss = F.mse_loss(pred, target)
            print(blue("[{0}: {1}/{2}] test loss: {3}".format(
                epoch, i, num_batch,
                loss.item(),
            )))

    torch.save(classifier.state_dict(), "{0}/model_{1}.pth".format(opt.outf, epoch))