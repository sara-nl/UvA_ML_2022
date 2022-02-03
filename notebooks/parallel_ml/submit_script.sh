#!/bin/bash -l
#SBATCH -J distrib_test
#SBATCH -o distrib_test_out.txt
#SBATCH -e distrib_test_err.txt

#SBATCH -t 0:10:00

#SBATCH --partition gpu_shared_course
#SBATCH --reservation jhlsrf011_overlap

#SBATCH --nodes 1
#SBATCH --gpus 2
#SBATCH --ntasks-per-node=2

#SBATCH --mem=64G

module load 2021
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1 torchvision/0.11.1-foss-2021a-CUDA-11.3.1

export OMP_NUM_THREADS=2

time python -m torch.distributed.launch --nproc_per_node=2 mnist_distrib.py