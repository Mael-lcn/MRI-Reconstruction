#!/bin/bash
#SBATCH --job-name=data_task
#SBATCH --qos=qos_cpu-t3
#SBATCH --cpus-per-task=16
#SBATCH --constraint=v100
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=3:00:00
#SBATCH --output=../dump/train.out
#SBATCH --error=../dump/train.err
#SBATCH --account=iql@v100

module purge
module load python/3.11.5
conda activate ig3d

srun python $WORK/ig3d/Cardiac-Volume-Segmentation/dataset/data_task.py -w 16
