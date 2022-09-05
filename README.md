# GPU Energy Benchmarking Framework

## Quickstart

This section gives a brief introduction, on how to get running with the benchmarking framework.

### Installation

First download, and import a cuda container from dockerhub.

```bash
enroot import docker://nvidia/cuda:11.7.1-runtime-ubuntu22.04
enroot create -n pyxis_cuda11 nvidia+cuda+11.7.1-runtime-ubuntu22.04.sqsh
```

Once you have your enroot container created, install some dependencies with the provided install script:

```bash
sbatch enroot/install-dependencies.sh
```

You can check the progress with `tail -f slurm-<job-id>.out`.

Once you have the dependencies installed, you can install the project itself.
Make sure to run the following script from the project root directory.
This script will create a new venv in the current directory, activate it and install the
project and its python dependencies in it.

```bash
sbatch enroot/install-project.sh
```

### Hello World

Once you installed all the requirements in your enroot container, you can test if everything works, by 
executing the hello world example:
```bash
./enroot/hello-world.sh
```

You can also launch an interactive session and run your own commands:
```bash
srun -A polze -p sorcery --container-name=cuda11 --container-writable --constraint="GPU_SKU:A100&GPU_MEM:20GB" --container-mount-home --container-workdir=$HOME --gpus=1 --pty bash
```

You can put the following alias in your `~/.bashrc`, then use it like `cuda11 --pty bash`.

```bash
alias cuda11='srun -A polze -p sorcery --container-name=cuda11 --container-writable --constraint="GPU_SKU:A100&GPU_MEM:20GB" --container-mount-home --container-workdir=$PWD --gpus=1 --export CONTAINER_NAME=cuda11'
```




## Project structure

- benchmarks: contains benchmarks scripts
  - 
- experiments: contains experiment definitions
- gpyjoules

