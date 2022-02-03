#!/bin/bash -l
#SBATCH -J distrib_test
#SBATCH -o distrib_test_out.txt
#SBATCH -e distrib_test_err.txt

#SBATCH -t 0:05:00

#SBATCH --partition gpu_shared
#SBATCH --reservation jhlsrf011_overlap

#SBATCH --nodes 1
#SBATCH --gpus 1
## SBATCH --gpus-per-node=1

#SBATCH --mem=64G

module load 2021
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1 torchvision/0.11.1-foss-2021a-CUDA-11.3.1
pip install tqdm

export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0

time python -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 mnist_distrib.py
