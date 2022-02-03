#!/bin/bash -l
#SBATCH -J distrib_test
#SBATCH -o distrib_test_out.txt
#SBATCH -e distrib_test_err.txt

#SBATCH -t 0:30:00

#SBATCH --partition gpu_shared_course
#SBATCH --reservation jupyterhub_course_jhlsrf011_2022-02-03

#SBATCH --nodes 1
#SBATCH --gpus 2
#SBATCH --ntasks-per-node=2


module load 2021
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1 torchvision/0.11.1-foss-2021a-CUDA-11.3.1

export OMP_NUM_THREADS=2

time python -m torch.distributed.launch --use_env --nproc_per_node=2 mnist_distrib.py