#!/bin/bash
#SBATCH -n 19
#SBATCH --mem-per-cpu=128G
#SBATCH -t 24:00:00
#SBATCH --array=0

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/xliu53/anaconda/ltl_transfer_v1/lib

module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate ltl_transfer_v1

python train_agent.py --algo ppo --env Simple-LTL-Env-v0 --ltl-sampler Transfer_train_hard_50 --log-interval 1 --save-interval 30 --procs 1 --frames 10000000 --epochs 2 --batch-size 1024 --frames-per-proc 512 --discount 0.9 --lr 0.001 --gae-lambda 0.5 --clip-eps 0.1 --dumb-ac
