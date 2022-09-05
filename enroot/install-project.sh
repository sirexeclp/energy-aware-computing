#!/bin/bash
#SBATCH -A polze
#SBATCH --partition sorcery

#SBATCH --container-name=cuda11
#SBATCH --container-writable

#SBATCH --container-mount-home
#SBATCH --container-workdir=$HOME

python3 -m venv venv
pip3 install .