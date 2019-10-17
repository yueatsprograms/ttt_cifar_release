from subprocess import call
import sys
import os

level = int(sys.argv[1])
shared = sys.argv[2]
setting = sys.argv[3]
name = sys.argv[4]

dataroot = '/home/yu/datasets/'		# PLEASE EDIT THIS

if level == 0:
	common_corruptions = ['cifar_new']
else:
	common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
	                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
	                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
	if level == 5:
		common_corruptions.append('original')

fix_bn = len(sys.argv) > 5 and sys.argv[5] == 'fix_bn'
fix_str = '_fix_bn' if fix_bn else ''
fix_tag = '--fix_bn' if fix_bn else ''

fix_ssh = len(sys.argv) > 5 and sys.argv[5] == 'fix_ssh'
fix_str = '_fix_ssh' if fix_ssh else ''
fix_tag = '--fix_ssh' if fix_ssh else ''
gpnorm_tag = '--group_norm 8' if name[:2] == 'gn' else ''
none_tag = '--none' if shared == 'none' else ''

if setting == 'fast':
	lr = 0.001
	niter = 1
	online_tag = ''
elif setting == 'medium':
	lr = 0.001
	niter = 3
	online_tag = ''
elif setting == 'slow':
	lr = 0.001
	niter = 10
	online_tag = ''
elif setting == 'jump':
	lr = 0.01
	niter = 1
	online_tag = ''
elif setting == 'online':
	lr = 0.001
	niter = 1
	online_tag = '--online'
	dset_size = 10000

batch_size_main = 128
batch_size_test = 32

for corruption in common_corruptions:
	print(corruption, level)
	call(' '.join(['python', 'test_calls/test_initial.py',
						dataroot,
						gpnorm_tag,
						none_tag,
						'--grad_corr',
						'--level 		%d' %(level),
						'--corruption	%s' %(corruption),
						'--shared 		%s' %(shared),
						'--batch_size	%d'	%(batch_size_main),
						'--resume 		results/cifar10_%s_%s/' %(shared, name),
						'--outf 		results/C10C_%s_%s_%s%s/' %(shared, setting, name, fix_str)]),
						shell=True)

	if shared == 'none':
		continue

	call(' '.join(['python', 'test_calls/show_decomp.py',
						'--level 		%d' %(level),
						'--corruption	%s' %(corruption),
						'--outf 		results/C10C_%s_%s_%s%s/' %(shared, setting, name, fix_str)]),
						shell=True)

	call(' '.join(['python', 'test_calls/test_adapt.py',
						dataroot,
						gpnorm_tag,
						online_tag,
						fix_tag,
						'--level 		%d' %(level),
						'--corruption	%s' %(corruption),
						'--shared 		%s' %(shared),
						'--batch_size	%d'	%(batch_size_test),
						'--lr 			%f' %(lr),
						'--niter		%d' %(niter),
						'--resume 		results/cifar10_%s_%s/' %(shared, name),
						'--outf 		results/C10C_%s_%s_%s%s/' %(shared, setting, name, fix_str)]),
						shell=True)

	call(' '.join(['python', 'test_calls/show_result.py',
						'--analyze_bin',
						'--analyze_ssh',
						'--level 		%d' %(level),
						'--corruption	%s' %(corruption),
						'--outf 		results/C10C_%s_%s_%s%s/' %(shared, setting, name, fix_str)]),
						shell=True)
