export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=0 python main.py --shared layer2 --rotation_type expand \
			--nepoch 150 --milestone_1 75 --milestone_2 125 \
			--outf results/cifar10_layer2_gn_expand

CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 layer2 online gn_expand

CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 4 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 4 layer2 online gn_expand

CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 3 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 3 layer2 online gn_expand

CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 2 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 2 layer2 online gn_expand

CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 1 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 1 layer2 online gn_expand

CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 0 layer2 slow gn_expand
CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 0 layer2 online gn_expand


CUDA_VISIBLE_DEVICES=0 python main.py --shared layer2 --rotation_type expand \
			--nepoch 75 --milestone_1 50 --milestone_2 65 \
			--outf results/cifar10_layer2_bn_expand

CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 layer2 slow bn_expand
CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 layer2 online bn_expand