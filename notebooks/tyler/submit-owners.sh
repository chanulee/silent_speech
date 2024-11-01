#!/bin/bash
#SBATCH -p owners
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --signal=B:SIGUSR1@90

. activate magneto
cd /home/users/tbenst/code/silent_speech/notebooks/tyler
srun python 2024-01-15_icml_models.py "$@"