Code release for [Test-Time Training for Out-of-Distribution Generalization](https://arxiv.org/abs/1909.13231).\
This code produces our results on CIFAR-10-C and CIFAR-10.1. 
The ImageNet results are produced by [this other repository](https://github.com/yueatsprograms/ttt_imagenet_release).

## Requirements
1. Our code requires pytorch version 1.0 or higher, with at least one modern GPU of adequate memory.
2. We ran our code with python 3.7. Compatibility with python 2 is possible maybe with some modifications.
3. Most of the packages used should be included with [anaconda](https://www.anaconda.com/distribution/), 
except maybe two small utilities:
	- [tqdm](https://github.com/tqdm/tqdm), which we installed with `conda install tqdm`.
	- [colorama](https://pypi.org/project/colorama/), which we installed with `conda install colorama`.
4. Download the two datasets into the same folder:
	- [CIFAR-10-C](https://arxiv.org/abs/1903.12261) (Hendrycks and Dietterich) 
from [this repository](https://github.com/hendrycks/robustness),
which links to [this shared storage](https://zenodo.org/record/2535967#.Xaf8uedKj-Y).
	- [CIFAR-10.1](https://arxiv.org/abs/1806.00451) (Recht et al.) 
from [this repository](https://github.com/modestyachts/CIFAR-10.1).

## Steps
1. Clone our repository with
`git clone https://github.com/yueatsprograms/ttt_cifar_release`.
2. Inside the repository, set the data folder to where the datasets are stored by editing:
	- `--dataroot` argument in `main.py`.
	- `dataroot` variable in `script_test_c10.py`.
3. Run `script.sh` for the main results, and `script_baseline.sh` for the baseline results.
4. The results are stored in the respective folders in `results/`.