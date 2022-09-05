#!/bin/bash
#SBATCH -A polze
#SBATCH --partition sorcery

#SBATCH --container-name=cuda11
#SBATCH --container-writable

#SBATCH --container-mount-home


cd $SLURM_SUBMIT_DIR
python3 -m venv venv
. venv/bin/activate
pip3 install .
echo "DONE."