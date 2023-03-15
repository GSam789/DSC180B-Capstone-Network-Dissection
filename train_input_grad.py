import numpy
import vgg16
import random
import time, datetime
import os, shutil
import yaml
import ast, bisect
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import grad
import torchnet as tnt
from dataloader import get_data



## CONDITIONS ##
epochs = 150
norm = 'L2'
lr = 0.01
momentum = 0.9
decay = 0.0005
penalty = 0.1
fd_order = 'O2'
## --------- ##

def scheduler(optimizer,lr_schedule):
    """Return a hyperparmeter scheduler for the optimizer"""
    lscheduler = LambdaLR(optimizer, lr_lambda = lr_schedule)

    return lscheduler

def test(epoch, ttot):
    model.eval()

    with torch.no_grad():

        # Get the true training loss and error
        top1_train = tnt.meter.ClassErrorMeter()
        train_loss = tnt.meter.AverageValueMeter()
        for data, target in train_loader:
            if has_cuda:
                target = target.cuda(0)
                data = data.cuda(0)

            output = model(data)


            top1_train.add(output.data, target.data)
            loss = criterion(output, target)
            train_loss.add(loss.data.item())

        t1t = top1_train.value()[0]
        lt = train_loss.value()[0]

        # Evaluate test data
        test_loss = tnt.meter.AverageValueMeter()
        top1 = tnt.meter.ClassErrorMeter()
        for data, target in test_loader:
            if has_cuda:
                target = target.cuda(0)
                data = data.cuda(0)

            output = model(data)

            loss = criterion(output, target)

            top1.add(output, target)
            test_loss.add(loss.item())

        t1 = top1.value()[0]
        l = test_loss.value()[0]

    print('[Epoch %2d] Average test loss: %.3f, error: %.2f%%'
            %(epoch, l, t1))
    print('%28s: %.3f, error: %.2f%%\n'
            %('training loss',lt,t1t))

    return test_loss.value()[0], top1.value()[0]


# --------
# Training
# --------

ix=0 #count of gradient steps

tik = penalty

regularizing = tik>0

h = 1 # finite difference step size

def train(epoch, ttot):
    global ix

    # Put the model in train mode (unfreeze batch norm parameters)
    model.train()

    # Run through the training data
    if has_cuda:
        torch.cuda.synchronize()
    tepoch = time.perf_counter()


    for batch_ix, (x, target) in enumerate(train_loader):

        if has_cuda:
            x = x.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        if regularizing:
            x.requires_grad_(True)

        prediction = model(x)
        lx = train_criterion(prediction, target)
        loss = lx.mean()


        # Compute finite difference approximation of directional derivative of grad loss wrt inputs
        if regularizing:

            dx = grad(loss, x, retain_graph=True)[0]
            sh = dx.shape
            x.requires_grad_(False)

            # v is the finite difference direction.
            # For example, if norm=='L2', v is the gradient of the loss wrt inputs
            v = dx.view(sh[0],-1)
            Nb, Nd = v.shape


            if norm=='L2':
                nv = v.norm(2,dim=-1,keepdim=True)
                nz = nv.view(-1)>0
                v[nz] = v[nz].div(nv[nz])
            if norm=='L1':
                v = v.sign()
                v = v/np.sqrt(Nd)
            elif norm=='Linf':
                vmax, Jmax = v.abs().max(dim=-1)
                sg = v.sign()
                I = torch.arange(Nb, device=v.device)
                sg = sg[I,Jmax]

                v = torch.zeros_like(v)
                I = I*Nd
                Ix = Jmax+I
                v.put_(Ix, sg)

            v = v.view(sh)
            xf = x + h*v

            mf = model(xf)
            lf = train_criterion(mf,target)
            if fd_order=='O2':
                xb = x - h*v
                mb = model(xb)
                lb = train_criterion(mb,target)
                H = 2*h
            else:
                H = h
                lb = lx
            dl = (lf-lb)/H # This is the finite difference approximation
                           # of the directional derivative of the loss


        tik_penalty = torch.tensor(np.nan)
        dlmean = torch.tensor(np.nan)
        dlmax = torch.tensor(np.nan)
        if tik>0:
            dl2 = dl.pow(2)
            tik_penalty = dl2.mean()/2
            loss = loss + tik*tik_penalty

        loss.backward()

        optimizer.step()

        if np.isnan(loss.data.item()):
            raise ValueError('model returned nan during training')

        t = ttot + time.perf_counter() - tepoch
        ix +=1

    if has_cuda:
        torch.cuda.synchronize()

    return ttot + time.perf_counter() - tepoch

def main():

    pct_max = 90.
    fail_count = fail_max = 5
    time = 0.
    pct0 = 100.
    for e in range(epochs):

        # Update the learning rate
        schedule.step()

        time = train(e, time)

        loss, pct_err= test(e,time)
        if pct_err >= pct_max:
            fail_count -= 1

        if pct_err < pct0:
            pct0 = pct_err

        if fail_count < 1:
            raise ValueError('Percent error has not decreased in %d epochs'%fail_max)


## LOAD DATA AND MODEL ##
train_loader, test_loader = get_data()
model = vgg16.VGG('VGG16')

## LOSS CONDITION ##
criterion = nn.CrossEntropyLoss()
train_criterion = nn.CrossEntropyLoss(reduction='none')

has_cuda = torch.cuda.is_available()
cudnn.benchmark = True
if has_cuda:
    criterion = criterion.cuda(0)
    train_criterion = train_criterion.cuda(0)
    model = model.cuda(0)

    
optimizer = optim.SGD(model.parameters(),
                  lr = lr,
                  weight_decay = decay,
                  momentum = momentum,
                  nesterov = False)

schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)

## RUN TRAINING ##
main()

## SAVE MODEL ##
filename = 'vgg16_cifar100_input_grad_reg_observed.pth'
torch.save(model.state_dict(), filename)


