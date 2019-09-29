from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torchdata

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import LinfPGDAttack

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='/home/yu/datasets/')
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=0, type=int)
########################################################################
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--nepoch', default=75, type=int)
parser.add_argument('--milestone_1', default=50, type=int)
parser.add_argument('--milestone_2', default=65, type=int)
########################################################################
parser.add_argument('--weight', default=0, type=float)
parser.add_argument('--alp', action='store_true')
########################################################################
parser.add_argument('--outf', default='.')

args = parser.parse_args()
import os
if os.path.isdir('/data/yusun/datasets/'):
    args.dataroot = '/data/yusun/datasets/'
elif os.path.isdir('/home/smartbuy/ssda/datasets/'):
    args.dataroot = '/home/smartbuy/ssda/datasets/'
elif os.path.isdir('/home/yu/datasets/'):
    args.dataroot = '/home/yu/datasets/'
elif os.path.isdir('/home/yusun/datasets/'):
    args.dataroot = '/home/yusun/datasets/'

args.shared = 'none'
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, _, _, _ = build_model(args)
_, teloader = prepare_test_data(args)
_, trloader = prepare_train_data(args)

optimizer = optim.SGD(list(net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
criterion = nn.CrossEntropyLoss().cuda()

all_err_cls = []
print('Running...')
print('Error (%)\t\ttest')

adversary = LinfPGDAttack(
    net, loss_fn=nn.CrossEntropyLoss().cuda(), eps=16/255,
    nb_iter=7, eps_iter=4/255, rand_init=True, clip_min=-1.0, clip_max=1.0,
    targeted=False)
if args.alp:
    criterion_alp = nn.MSELoss().cuda()

for epoch in range(1, args.nepoch+1):
    net.train()

    for batch_idx, (inputs, labels) in enumerate(trloader):
        inputs_cls, labels_cls = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        with ctx_noparamgrad_and_eval(net):
            inputs_adv = adversary.perturb(inputs_cls, labels_cls)

        if args.weight == 0:
            outputs_adv = net(inputs_adv)
            loss = criterion(outputs_adv, labels_cls)
        else:
            inputs_all = torch.cat([inputs_cls, inputs_adv], dim=0)
            labels_all = torch.cat([labels_cls, labels_cls], dim=0)
            outputs_all = net(inputs_all)
            outputs_cls, outputs_adv = torch.split(outputs_all, inputs_cls.size(0), dim=0)
            loss = criterion(outputs_cls, labels_cls)

            if args.alp:
                loss += args.weight * criterion_alp(outputs_cls, outputs_adv)
            else:
                loss += args.weight * criterion(outputs_adv, labels_cls)

        loss.backward()
        optimizer.step()

    err_cls, _, _ = test(teloader, net)
    all_err_cls.append(err_cls)
    scheduler.step()

    print(('Epoch %d/%d:' %(epoch, args.nepoch)).ljust(24) + '%.2f' %(err_cls*100))
    torch.save((all_err_cls), args.outf + '/loss.pth')
    plot_epochs(all_err_cls, all_err_cls, args.outf + '/loss.pdf', use_agg=True)
        
state = {'err_cls': err_cls, 'optimizer': optimizer.state_dict(), 'net': net.state_dict()}
torch.save(state, args.outf + '/ckpt.pth')
