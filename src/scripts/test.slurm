#!/bin/bash
#SBATCH --job-name=test_diffcmr
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --account=iql@v100
#SBATCH --output=../dump/test.out
#SBATCH --error=../dump/test.err

module purge
module load pytorch-gpu/py3/2.1.1
conda activate ig3d

# Chemin vers tes logs d'inférence
export OPENAI_LOGDIR="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/flow/eval_final"

# Lancement distribué
# Note : On utilise les arguments définis dans le create_argparser() de ton script python
srun python $WORK/ig3d/Cardiac-Volume-Segmentation/src/DiffCMR/inference.py \
    --model_path "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/flow/logs_run1/model44000.pt" \
    --val_pair_file "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/ValidationSet/pairs.txt" \
    --result_dir "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/flow/eval/" \
    --diffusion_steps 30 \
    --batch_size 8 \
    --vote_num 1 \
    --num_workers 2
