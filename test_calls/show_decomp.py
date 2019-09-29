import argparse
import numpy as np
import torch

def plot_losses(cls_losses, ssh_losses, fname, use_agg=True):
	from utils.misc import normalize
	import matplotlib.pyplot as plt
	if use_agg:
		plt.switch_backend('agg')

	colors = ['r', 'g', 'b', 'm']
	labels = range(4)
	cls_losses = normalize(cls_losses)
	for losses, color, label in zip(ssh_losses, colors, labels):
		losses = normalize(losses)
		plt.scatter(cls_losses, losses, label=str(label), color=color, s=4)
		plt.xlabel('classification loss')
		plt.ylabel('rotation loss')
		plt.savefig('%s_scatter_%d.pdf' %(fname, label))
		plt.close()


def decomp_rand(clse, sshe, total):
	clsw = total * clse
	clsr = total - clsw

	crr = clsr * (1-sshe)
	crw = clsr * sshe
	cwr = clsw * (1-sshe)
	cww = clsw * sshe
	return int(crr), int(crw), int(cwr), int(cww)

def show_decomp(cls_initial, cls_correct, all_ssh_initial, all_ssh_correct, fname, use_agg=False):
	if use_agg:
		import matplotlib
		matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from utils.test_helpers import count_each, pair_buckets

	labels = range(4)
	for ssh_initial, ssh_correct, label in zip(all_ssh_initial, all_ssh_correct, labels):
		print('Direction %d error %.2f' %(label, ssh_initial*100))

		dtrue = count_each(pair_buckets(cls_correct, ssh_correct))
		torch.save(dtrue, '%s_dec_%d.pth' %(fname, label))
		print('Error decoposition:', *dtrue)
		drand = decomp_rand(cls_initial, ssh_initial, sum(dtrue))	

		width = 0.25
		ind = np.arange(4)
		plt.bar(ind, 		drand, width, label='independent')
		plt.bar(ind+width, 	dtrue, width, label='observed')

		plt.ylabel('count for label %d' %(label))
		plt.xticks(ind + width/2, ('RR', 'RW', 'WR', 'WW'))
		plt.legend(loc='best')
		plt.savefig('%s_bar_%d.pdf' %(fname, label))
		plt.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--level', default=0, type=int)
	parser.add_argument('--corruption', default='original')
	parser.add_argument('--outf', default='.')
	args = parser.parse_args()
	rdict = torch.load(args.outf + '/%s_%d_inl.pth' %(args.corruption, args.level))
	fname = args.outf + '/%s_%d' %(args.corruption, args.level)

	plot_losses(rdict['cls_losses'], rdict['ssh_losses'], fname, use_agg=True)
	show_decomp(rdict['cls_initial'], rdict['cls_correct'],
				rdict['ssh_initial'], rdict['ssh_correct'], fname, use_agg=True)
