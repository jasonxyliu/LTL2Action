#!/bin/bash
#SBATCH -n 16
#SBATCH --mem=9G
#SBATCH -t 1:00:00
#SBATCH --array=0

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate ltl_transfer_v1

python train_agent.py --algo ppo --env Simple-LTL-Env-v0 --ltl-sampler Transfer_train_mixed_50 --log-interval 1 --save-interval 30 --frames 10000000 --epochs 2 --batch-size 1024 --frames-per-proc 512 --discount 0.9 --lr 0.001 --gae-lambda 0.5 --clip-eps 0.1 --dumb-ac
