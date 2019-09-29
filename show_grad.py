import numpy as np
import torch
from utils.misc import *
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import seaborn as sns
sns.set_palette('colorblind')

levels = [5, 4, 3, 2, 1]
common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

folders = []
folders.append('results/C10C_layer2_slow_gn_expand_final')
folders.append('results/C10C_layer2_online_gn_expand_final')

for folder in folders:
	plt.figure(figsize=(4, 3))

	all_x = []
	all_y = []
	xdata = []
	ydata = []
	for level in levels:
		xdata = []
		ydata = []
		print(level)
		for corruption in common_corruptions:
			xdata_local = torch.load(folder + '/%s_%d_grc.pth' %(corruption, level))
			xdata_local = mean(xdata_local)

			rdict_ada = torch.load(folder + '/%s_%d_ada.pth' %(corruption, level))
			rdict_inl = torch.load(folder + '/%s_%d_inl.pth' %(corruption, level))
			new_correct = rdict_ada['cls_adapted']
			old_correct = rdict_inl['cls_initial']
			ydata_local = (old_correct - new_correct) * 100

			xdata.append(xdata_local)
			ydata.append(ydata_local)

		plt.scatter(xdata, ydata, s=15, label='Level %d' %(level))
		all_x += xdata
		all_y += ydata

	correlation = pearsonr(all_x, all_y)
	print('Pearson correlation coefficient: %.3f' %(correlation[0]))
	import seaborn as sns
	sns.regplot(all_x, all_y, color='blue', ci=99, scatter=False)

	plt.xlabel('Gradient inner product')
	plt.ylabel('Improvement (%)')
	plt.legend(prop={'size': 8})
	plt.tight_layout()
	plt.savefig(folder + '_grad_corr.pdf')
	plt.close()
