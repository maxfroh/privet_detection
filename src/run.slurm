#!/bin/bash
#SBATCH --job-name=b2_lr0.1_e100
#SBATCH --partition=gpu-a100
#SBATCH --output=b2_lr0.1_e100.%j.out
#SBATCH --error=b2_lr0.1_e100.%j.err
#SBATCH --account=CCR24017
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT_50,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=mbf1102@rit.edu

set -e
cd $SLURM_SUBMIT_DIR

srun -N1 -n1 --exclusive bash -c "source /scratch/10746/maxfroh/envs/privet/bin/activate && python /scratch/10746/maxfroh/privet_detection/src/main.py --model=faster_rcnn --channels=rgb --num_epochs=100 --batch_size=2 --learning_rate=0.1 --results_dir=/scratch/10746/maxfroh/data/results --img_dir=/scratch/10746/maxfroh/data/images --labels_dir=/scratch/10746/maxfroh/data/labels --scheduler_step_size=1 --scheduler_gamma=0.1 --kfold=5" &

wait

