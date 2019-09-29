export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=0 python main.py \
			--nepoch 150 --milestone_1 75 --milestone_2 125 \
			--outf results/cifar10_none_gn
for i in $(seq 1 5);
do CUDA_VISIBLE_DEVICES=0 python script_test_c10.py $i none none gn
done
CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 0 none none gn


CUDA_VISIBLE_DEVICES=0 python main.py \
			--nepoch 75 --milestone_1 50 --milestone_2 65 \
			--outf results/cifar10_none_bn
CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 none none bn


CUDA_VISIBLE_DEVICES=0 python baseline.py --nepoch 150 --milestone_1 75 --milestone_2 125 --group_norm 8 \
											 --alp --weight 1 --outf results/cifar10_none_gn_bl_1_alp
CUDA_VISIBLE_DEVICES=0 python baseline.py --nepoch 150 --milestone_1 75 --milestone_2 125 --group_norm 8 \
											 --alp --weight 0.5 --outf results/cifar10_none_gn_bl_0.5_alp

for i in $(seq 1 5);
do CUDA_VISIBLE_DEVICES=0 python script_test_c10.py $i none baseline gn_bl_0.5_alp;
done
for i in $(seq 1 5);
do CUDA_VISIBLE_DEVICES=0 python script_test_c10.py $i none baseline gn_bl_1_alp;
done