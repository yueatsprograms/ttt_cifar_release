import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize(*NORM)])
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize(*NORM)])
mnist_transforms = transforms.Compose([transforms.Resize((32, 32)),
										transforms.ToTensor(),
										transforms.Normalize((0.1307,), (0.3081,))])

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
					'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
					'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def prepare_test_data(args):
	if args.dataset == 'cifar10':
		tesize = 10000
		if not hasattr(args, 'corruption') or args.corruption == 'original':
			print('Test on the original test set')
			teset = torchvision.datasets.CIFAR10(root=args.dataroot,
												train=False, download=True, transform=te_transforms)
		elif args.corruption in common_corruptions:
			print('Test on %s level %d' %(args.corruption, args.level))
			teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
			teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
			teset = torchvision.datasets.CIFAR10(root=args.dataroot,
												train=False, download=True, transform=te_transforms)
			teset.data = teset_raw

		elif args.corruption == 'cifar_new':
			from utils.cifar_new import CIFAR_New
			print('Test on CIFAR-10.1')
			teset = CIFAR_New(root=args.dataroot + 'CIFAR-10.1/datasets/', transform=te_transforms)
			permute = False
		else:
			raise Exception('Corruption not found!')
	else:
		raise Exception('Dataset not found!')

	if not hasattr(args, 'workers'):
		args.workers = 1
	teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
											shuffle=False, num_workers=args.workers)
	return teset, teloader

def prepare_train_data(args):
	print('Preparing data...')
	if args.dataset == 'cifar10':
		trset = torchvision.datasets.CIFAR10(root=args.dataroot,
										train=True, download=True, transform=tr_transforms)
	else:
		raise Exception('Dataset not found!')

	if not hasattr(args, 'workers'):
		args.workers = 1
	trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
											shuffle=True, num_workers=args.workers)
	return trset, trloader
	