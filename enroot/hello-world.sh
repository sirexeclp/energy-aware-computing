srun \
    -A polze \
    --partition sorcery \
    --export=HOME,PWD \
    --container-name=cuda11 \
    --container-mount-home \
    --container-workdir=$PWD \
    --gpus=1 \
    --constraint="GPU_SKU:A100&GPU_MEM:20GB" \
    bash -c ". venv/bin/activate && python3 -m gpyjoules.info"