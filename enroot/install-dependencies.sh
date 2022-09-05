#!/bin/bash
#SBATCH -A polze
#SBATCH --partition sorcery

#SBATCH --container-name=cuda11
#SBATCH --container-writable
#SBATCH --container-remap-root

apt update
apt install -y python3 python3-venv python3-pip git micro
echo "DONE."