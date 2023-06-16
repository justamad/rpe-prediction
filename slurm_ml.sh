#!/bin/bash
#SBATCH --job-name=default
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin.albert@hpi.de
#SBATCH --partition=hpcpu,vcpu # -p
#SBATCH --cpus-per-task=64 # -c
#SBATCH --mem=200gb
#SBATCH --time=72:00:00
#SBATCH --output=job_test_%j.log # %j is job id

# YAIB_PATH=/dhc/home/robin.vandewater/projects/yaib #/dhc/home/robin.vandewater/projects/yaib
# cd ${YAIB_PATH}

eval "$(conda shell.bash hook)"
conda activate rpe-prediction

# DATASET_ROOT_PATH=data/YAIB_Datasets/data #data/YAIB_Datasets/data
# DATASETS=(aumc hirid eicu miiv)

echo "This is a SLURM job named" $SLURM_JOB_NAME
htop

python3 train_ml.py
