#!/bin/bash

dataset=$1

# Run the code below in command window to set CUDA visible devices and run specific script
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#sh run_sid.sh 'cifar10-uncond' 

# Modify --duration to reproduce the reported results
# 
# As explicitly stated in the paper and illustrated in the figures, the default setting for 
# reproducing the reported results was a duration of 500 million (--duration 500), except for 
# CIFAR-10 conditional under alpha = 1.2, where it was increased to 800, and ImageNet 64×64 
# under alpha = 1.2, where it was raised to 1000. Additionally, we have released all our 
# checkpoints for further verification. Setting --duration 100 will produce very competitive 
# results, as shown in Figures 2–6, but will not match the numbers reported in Tables 1–4.
# 
# Decrease --batch-gpu to reduce memory consumption

if [ "$dataset" = 'cifar10-uncond' ]; then
    # Command to run torch with specific parameters
    # Add the option below to load a checkpoint:
    #Many options are optional, such as --data_stat, which will be computed inside the code if not provided
    # --resume 'image_experiment/sid-train-runs/cifar10-uncond/training-state-????.pt'
    # python sid_train.py \
    torchrun --standalone --nproc_per_node=2 sid_train.py \
    --alpha 1.2 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --outdir 'image_experiment/sid-train-runs/cifar10-uncond' \
    --data 'data/datasets/cifar10-32x32.zip' \
    --arch ddpmpp \
    --edm_model cifar10-uncond \
    --metrics fid50k_full,is50k \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 0 \
    --ls 1 \
    --lsg 100 \
    --duration 500 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz'
    #\
    #--resume 'image_experiment/sid-train-runs/cifar10-uncond/00002-cifar10-32x32-uncond-ddpmpp-edmglr1e-05-lr1e-05-initsigma2.5-gpus4-alpha1.2-batch256-tmax800-fp32/training-state-000128.pt'
else
    echo "Invalid dataset specified"
    exit 1
fi
