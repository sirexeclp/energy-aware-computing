#!/bin/bash
#SBATCH -A polze
#SBATCH --partition sorcery

#SBATCH --container-name=cuda11
#SBATCH --container-writable

#SBATCH --container-mount-home
#SBATCH --container-workdir=$PWD

#SBATCH --gpus=1
#SBATCH --constraint="GPU_SKU:A100&GPU_MEM:20GB"

python3 -m gpyjoules.info