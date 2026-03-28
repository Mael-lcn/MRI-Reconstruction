#!/bin/bash
#SBATCH --job-name=train_diffcmr
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --account=iql@v100
#SBATCH --output=../dump/train.out
#SBATCH --error=../dump/train.err

module purge
module load pytorch-gpu/py3/2.1.1
conda activate ig3d

export OPENAI_LOGDIR="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/flow/logs_run1"

# Lancement distribué
srun python $WORK/ig3d/Cardiac-Volume-Segmentation/src/DiffCMR/train_task1_flow.py \
    --trainpairfile "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/TrainingSet/pairs.txt" \
    --logdir "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/flow/logs_run1" \
    --use_fp16 True \
    --num_workers 2 \
    --batch_size 64 \
    --microbatch 16 \
    --run_without_test False \
    --log_interval 50 \
    --save_interval 2000 \
    --start_print_iter 0 \
    --resume_checkpoint "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/flow/logs_run1"
