import sys
import numpy as np
import torch
from utils.misc import *
from test_calls.show_result import get_err_adapted

corruptions_names = ['gauss', 'shot', 'impulse', 'defocus', 'glass', 'motion', 'zoom', 
							'snow', 'frost', 'fog', 'bright', 'contra', 'elastic', 'pixel', 'jpeg']
corruptions_names.insert(0, 'orig')

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
					'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
					'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
corruptions.insert(0, 'original')

info = []
info.append(('gn', '_expand_final', 5))
info.append(('gn', '_expand_final', 4))
info.append(('gn', '_expand_final', 3))
info.append(('gn', '_expand_final', 2))
info.append(('gn', '_expand_final', 1))
info.append(('bn', '_expand_final', 5))

for level in [1,2,3,4,5]:
	baseline += [('', '', level)]
	baseline += [('gn', '1_alp', level)]
	baseline += [('gn', '0.5_alp', level)]

########################################################################

def print_table(table, prec1=True):
	for row in table:
		row_str = ''
		for entry in row:
			if prec1:
				row_str += '%.1f\t' %(entry)
			else:
				row_str += '%s\t' %(str(entry))
		print(row_str)

def show_table(folder, level, threshold):
	results = []
	for corruption in corruptions:
		row = []
		try:
			rdict_ada = torch.load(folder + '/%s_%d_ada.pth' %(corruption, level))
			rdict_inl = torch.load(folder + '/%s_%d_inl.pth' %(corruption, level))

			ssh_confide = rdict_ada['ssh_confide']
			new_correct = rdict_ada['cls_correct']
			old_correct = rdict_inl['cls_correct']

			row.append(rdict_inl['cls_initial'])
			old_correct = old_correct[:len(new_correct)]
			err_adapted = get_err_adapted(new_correct, old_correct, ssh_confide, threshold=threshold)
			row.append(err_adapted)

		except:
			row.append(0)
			row.append(0)
		results.append(row)

	results = np.asarray(results)
	results = np.transpose(results)
	results = results * 100
	return results

def show_none(folder, level):
	results = []
	for corruption in corruptions:
		try:
			rdict_inl = torch.load(folder + '/%s_%d_none.pth' %(corruption, level))
			results.append(rdict_inl['cls_initial'])
		except:
			results.append(0)
	results = np.asarray([results])
	results = results * 100
	return results

for parta, partb, level in info:
	print(level, parta + partb)
	print_table([corruptions_names], prec1=False)
	if parta == 'bn':
		threshold = 0.9
	else:
		threshold = 1

	results_none = show_none('results/C10C_none_%s_%s' %('none', parta), level)
	print_table(results_none)

	results_slow = show_table('results/C10C_layer2_%s_%s%s' %('slow', parta, partb), level, threshold=threshold)
	print_table(results_slow)

	results_onln = show_table('results/C10C_layer2_%s_%s%s' %('online', parta, partb), level, threshold=threshold)
	results_onln = results_onln[1:,:]
	print_table(results_onln)

	results = np.concatenate((results_none, results_slow, results_onln))
	torch.save(results, 'results/C10C_layer2_%d_%s%s.pth' %(level, parta, partb))

for parta, partb, level in baseline:
	if parta == '':
		print(level)
		print_table([corruptions_names], prec1=False)
		continue
	results_none = show_none('results/C10C_none_baseline_%s_bl_%s' %(parta, partb), level)
	print_table(results_none)
