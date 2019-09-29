from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from utils.rotation import rotate_batch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='/home/yu/datasets/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=0, type=int)
########################################################################
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--nepoch', default=75, type=int)
parser.add_argument('--milestone_1', default=50, type=int)
parser.add_argument('--milestone_2', default=65, type=int)
parser.add_argument('--rotation_type', default='rand')
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

my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)
_, teloader = prepare_test_data(args)
_, trloader = prepare_train_data(args)

parameters = list(net.parameters())+list(head.parameters())
optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
criterion = nn.CrossEntropyLoss().cuda()

all_err_cls = []
all_err_ssh = []
print('Running...')
print('Error (%)\t\ttest\t\tself-supervised')
for epoch in range(1, args.nepoch+1):
    net.train()
    ssh.train()

    for batch_idx, (inputs, labels) in enumerate(trloader):
        optimizer.zero_grad()
        inputs_cls, labels_cls = inputs.cuda(), labels.cuda()
        outputs_cls = net(inputs_cls)
        loss = criterion(outputs_cls, labels_cls)

        if args.shared is not None:
            inputs_ssh, labels_ssh = rotate_batch(inputs, args.rotation_type)
            inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
            outputs_ssh = ssh(inputs_ssh)
            loss_ssh = criterion(outputs_ssh, labels_ssh)
            loss += loss_ssh

        loss.backward()
        optimizer.step()

    err_cls = test(teloader, net)[0]
    err_ssh = 0 if args.shared is None else test(teloader, ssh, sslabel='expand')[0]
    all_err_cls.append(err_cls)
    all_err_ssh.append(err_ssh)
    scheduler.step()

    print(('Epoch %d/%d:' %(epoch, args.nepoch)).ljust(24) +
                    '%.2f\t\t%.2f' %(err_cls*100, err_ssh*100))
    torch.save((all_err_cls, all_err_ssh), args.outf + '/loss.pth')
    plot_epochs(all_err_cls, all_err_ssh, args.outf + '/loss.pdf')

state = {'err_cls': err_cls, 'err_ssh': err_ssh,
            'net': net.state_dict(), 'head': head.state_dict(),
            'optimizer': optimizer.state_dict()}
torch.save(state, args.outf + '/ckpt.pth')
