#!/bin/bash
#SBATCH --job-name=default
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin.albert@hpi.de
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=16 # -c
#SBATCH --mem=10gb
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=job_test_%j.log # %j is job id

# YAIB_PATH=/dhc/home/robin.vandewater/projects/yaib #/dhc/home/robin.vandewater/projects/yaib
# cd ${YAIB_PATH}

eval "$(conda shell.bash hook)"
conda activate tf

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib


# DATASET_ROOT_PATH=data/YAIB_Datasets/data #data/YAIB_Datasets/data
# DATASETS=(aumc hirid eicu miiv)

echo "This is a SLURM job named" $SLURM_JOB_NAME
htop

python3 train_dl.py
# python3 test.py
