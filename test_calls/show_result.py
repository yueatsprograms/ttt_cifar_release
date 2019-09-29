import argparse
import numpy as np
import torch
from utils.misc import *
from utils.test_helpers import pair_buckets

def analyze(idx_tbd, idx_all, err):
	new_tbd = np.logical_and(idx_all, idx_tbd).sum()
	new_per = new_tbd.sum() / idx_tbd.sum()
	if err:
		print_color('RED', 		'%d\t%d\t%.2f' %(idx_tbd.sum(), idx_tbd.sum() - new_tbd.sum(), (1-new_per)*100))
	else:
		print_color('GREEN',	'%d\t%d\t%.2f' %(idx_tbd.sum(), new_tbd.sum(), new_per*100))

def analyze_all(adapted, all_initial):
	errs = [True, True, False, False]
	for err, initial in zip(errs, all_initial):
		analyze(initial, adapted, err)

def show_result(adapted, initial):
	print('Error (%)')
	print_color('RED', 		'%.1f' %(initial*100))
	print_color('YELLOW', 	'%.1f' %(adapted*100))
	print_color('GREEN',	'%.1f' %((initial - adapted)*100))

def get_err_adapted(new_correct, old_correct, ssh_confide, threshold=1):
	adapted = new_correct[ssh_confide < threshold]
	noadptd	= old_correct[ssh_confide >= threshold]
	# print('Percentage adapted: %.1f' % (len(adapted) / len(new_correct) * 100))
	return 1 - (sum(adapted) + sum(noadptd)) / (len(adapted) + len(noadptd))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--level', default=0, type=int)
	parser.add_argument('--corruption', default='original')
	parser.add_argument('--outf', default='.')
	parser.add_argument('--threshold', default=1, type=float)
	parser.add_argument('--dset_size', default=0, type=int)
	parser.add_argument('--analyze_bin', action='store_true')
	parser.add_argument('--analyze_ssh', action='store_true')
	parser.add_argument('--save_oh', action='store_true')

	args = parser.parse_args()
	args.threshold += 0.001		# to correct for numeric errors
	rdict_ada = torch.load(args.outf + '/%s_%d_ada.pth' %(args.corruption, args.level))
	rdict_inl = torch.load(args.outf + '/%s_%d_inl.pth' %(args.corruption, args.level))

	ssh_confide = rdict_ada['ssh_confide']
	new_correct = rdict_ada['cls_correct']
	old_correct = rdict_inl['cls_correct']

	if args.dset_size == 0:
		args.dset_size = len(old_correct)

	old_correct = old_correct[:args.dset_size]
	err_adapted = get_err_adapted(new_correct, old_correct, ssh_confide, threshold=args.threshold)
	show_result(err_adapted, 1-mean(old_correct))

	if args.analyze_bin:
		print('Bin analysis')
		for label, ssh_correct in enumerate(rdict_inl['ssh_correct']):
			ssh_correct = ssh_correct[:args.dset_size]
			dvecs = pair_buckets(old_correct, ssh_correct)
			print('Direction %d' %(label))
			analyze_all(rdict_ada['cls_correct'], dvecs)

	if args.analyze_ssh:
		print('Self-supervised analysis')
		for label, ssh_correct in enumerate(rdict_inl['ssh_correct']):
			ssh_correct = ssh_correct[:args.dset_size]
			trerror = 1 - mean([correct[label].item() for correct in rdict_ada['trerror']])
			print('Direction %d' %(label))
			print(' Old error (%%): %.2f' %((1-mean(ssh_correct)) * 100))
			print(' New error (%%): %.2f' %(trerror * 100))

	if args.save_oh:
		torch.save((old_correct, new_correct), args.outf + '/one_hot_saved.pth')
