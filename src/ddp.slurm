#!/bin/bash
#SBATCH --job-name=ddp_test
#SBATCH --partition=gpu-a100
#SBATCH --output=ddp_test.%j.out
#SBATCH --error=ddp_test.%j.err
#SBATCH --account=CCR24017
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT_50,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=mbf1102@rit.edu

set -e
cd $SLURM_SUBMIT_DIR

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=29500
export WORLD_SIZE=3
export RANK=$SLURM_LOCALID

srun -N1 -n1 --exclusive bash -c "source /scratch/10746/maxfroh/envs/privet/bin/activate && python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=$SLURM_LOCALID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT /scratch/10746/maxfroh/privet_detection/src/main.py --model=faster_rcnn --channels=rgb --num_epochs=4 --batch_size=2 --learning_rate=0.1 --results_dir=/scratch/10746/maxfroh/data/results --img_dir=/scratch/10746/maxfroh/data/images --labels_dir=/scratch/10746/maxfroh/data/labels --scheduler_step_size=1 --scheduler_gamma=0.1 --kfold=5" &

wait

