uv run python sid_generate.py \
    --outdir=image_experiment/out \
    --seeds=0-49999 \
    --batch=32 \
    --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-cond/alpha1.2/network-snapshot-1.200000-713312.pkl' \
    --ref='https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' \
    --aug_dim=1 \
    --single_batch=True
