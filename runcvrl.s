#!/bin/bash
#
#SBATCH --job-name=cvrlGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

cd /ibex/scratch/$USER/cvrl

nvidia-smi

module purge
module load python/intel/3.8.6

pip install torch
pip install torchvision
pip install tqdm
pip install thop

python train_model.py --model_name mocov1 --batch_size 512 --epochs 200 --arch resnet18 --learning_rate 0.06 --temperature 0.1 --weight_decay 5e-4