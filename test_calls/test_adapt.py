from __future__ import print_function
import argparse
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from utils.rotation import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--dataroot', default='/data/yusun/datasets/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--fix_bn', action='store_true')
parser.add_argument('--fix_ssh', action='store_true')
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--dset_size', default=0, type=int)
########################################################################
parser.add_argument('--outf', default='.')
parser.add_argument('--resume', default=None)

args = parser.parse_args()
args.threshold += 0.001		# to correct for numeric errors
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)
teset, teloader = prepare_test_data(args)

print('Resuming from %s...' %(args.resume))
ckpt = torch.load(args.resume + '/ckpt.pth')
if args.online:
	net.load_state_dict(ckpt['net'])
	head.load_state_dict(ckpt['head'])

criterion_ssh = nn.CrossEntropyLoss().cuda()
if args.fix_ssh:
	optimizer_ssh = optim.SGD(ext.parameters(), lr=args.lr)
else:
	optimizer_ssh = optim.SGD(ssh.parameters(), lr=args.lr)

def adapt_single(image):
	if args.fix_bn:
		ssh.eval()
	elif args.fix_ssh:
		ssh.eval()
		ext.train()
	else:
		ssh.train()
	for iteration in range(args.niter):
		inputs = [tr_transforms(image) for _ in range(args.batch_size)]
		inputs = torch.stack(inputs)
		inputs_ssh, labels_ssh = rotate_batch(inputs, 'rand')
		inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
		optimizer_ssh.zero_grad()
		outputs_ssh = ssh(inputs_ssh)
		loss_ssh = criterion_ssh(outputs_ssh, labels_ssh)
		loss_ssh.backward()
		optimizer_ssh.step()

def test_single(model, image, label):
	model.eval()
	inputs = te_transforms(image).unsqueeze(0)
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)
		confidence = nn.functional.softmax(outputs, dim=1).squeeze()[label].item()
	correctness = 1 if predicted.item() == label else 0
	return correctness, confidence

def trerr_single(model, image):
	model.eval()
	labels = torch.LongTensor([0, 1, 2, 3])
	inputs = torch.stack([te_transforms(image) for _ in range(4)])
	inputs = rotate_batch_with_labels(inputs, labels)
	inputs, labels = inputs.cuda(), labels.cuda()
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)
	return predicted.eq(labels).cpu()

print('Running...')
correct = []
sshconf = []
trerror = []
if args.dset_size == 0:
	args.dset_size = len(teset)
for i in tqdm(range(1, args.dset_size+1)):
	if not args.online:
		net.load_state_dict(ckpt['net'])
		head.load_state_dict(ckpt['head'])

	_, label = teset[i-1]
	image = Image.fromarray(teset.data[i-1])

	sshconf.append(test_single(ssh, image, 0)[1])
	if sshconf[-1] < args.threshold:
		adapt_single(image)
	correct.append(test_single(net, image, label)[0])
	trerror.append(trerr_single(ssh, image))

rdict = {'cls_correct': np.asarray(correct), 'ssh_confide': np.asarray(sshconf), 
		'cls_adapted':1-mean(correct), 'trerror': trerror}
torch.save(rdict, args.outf + '/%s_%d_ada.pth' %(args.corruption, args.level))
