cd $(dirname $0)/src
echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
nvidia-smi
sudo --preserve-env=NVIDIA_VISIBLE_DEVICES python3 -m run_benchmark
