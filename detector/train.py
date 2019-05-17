from __future__ import print_function
import argparse
import os
import random
import utils
import evaluation
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
from detector import PointNetDenseCls4D as net
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
parser.add_argument('--npoints', type=int, default=25000, help='number of points to sample')
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
num_classes = dataset.nclasses
print("# of    object classes: {0}".format(num_classes))
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = net(num_points=opt.npoints, num_classes=num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.SGD(classifier.parameters(), lr=0.0001, momentum=0.9)
classifier.cuda()

focal_loss = FocalLoss(opt.gamma)
focal_loss = focal_loss.cuda()

# 0.15 for background, 0.85 for liquid
# [background, liquid]
loss_weight = torch.Tensor([1.0, 3.0]).cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target, _, _ = data
        points, target = Variable(points), Variable(target)
        points = points.transpose(2, 1)

        points, target = points.cuda(), target.cuda()
        #points = points.cuda()
        optimizer.zero_grad()
        pred, _ = classifier(points)
        pred = pred.view(-1, num_classes)
        pred_choice = pred.data.max(1)[1]
        target = target.view(-1)
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target, weight=loss_weight)
        #loss = focal_loss(pred, target)
        loss.backward()
        optimizer.step()
        prediction = pred_choice.cpu().numpy()
        target = target.data.cpu().numpy()
        iou = evaluation.compute_iou(prediction, target)
        #correct = pred_choice.eq(target.data).cpu().sum()
        #correct = pred_choice.eq(target.data).sum()
        print("[{0}: {1}/{2}] train loss: {3} IoU: {4}".format(
            epoch, i, num_batch,
            loss.item(),
            iou
            #correct.item() / float(opt.batchSize * opt.npoints)
        ))
        if i % 32 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target, _, _ = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            #points = points.cuda()
            pred, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            pred_choice = pred.data.max(1)[1]
            target = target.view(-1)

            loss = F.nll_loss(pred, target, weight=loss_weight)
            prediction = pred_choice.cpu().numpy()
            target = target.data.cpu().numpy()
            iou = evaluation.compute_iou(prediction, target)
            #pred = pred.cpu()
            #loss = focal_loss(pred, target)
            #correct = pred_choice.eq(target.data).cpu().sum()
            #correct = pred_choice.eq(target.data).sum()
            print(blue("[{0}: {1}/{2}] test loss: {3} IoU: {4}".format(
                epoch, i, num_batch,
                loss.item(),
                iou
                #correct.item() / float(opt.batchSize * opt.npoints)
            )))

    torch.save(classifier.state_dict(), "{0}/model_{1}.pth".format(opt.outf, epoch))
