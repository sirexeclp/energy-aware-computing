#!/bin/bash
#SBATCH -A polze
#SBATCH --partition sorcery

#SBATCH --container-name=cuda11
#SBATCH --container-writable
#SBATCH --container-remap-root

##SBATCH --container-mount-home
##SBATCH --container-workdir=$HOME

##SBATCH --gpus=1
##SBATCH --constraint="GPU_SKU:A100&GPU_MEM:20GB"

apt update
apt install -y python3 python3-venv python3-pip git micro