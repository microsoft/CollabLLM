#!/bin/bash

# Source shared parameters
source scripts/config.sh

# Ensure a dataset name is provided as input
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

# Set dataset-specific parameters, start with base model
set_dataset_config "$1"
set_assistant_model "$1" "base"

RANDOM_SEED=$$
PORT=$((56400 + RANDOM_SEED % 10))
export TRITON_CACHE_DIR=$OUTPUT_DIR/cache

CUDA_VISIBLE_DEVICES=2,3,4,5 WANDB__SERVICE_WAIT=300 torchrun --master_port=$PORT --nnodes=1 --nproc_per_node=4 \
    scripts/sft_train.py \
    --dataset org_name/collabllm-$DATASET \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --max_new_tokens $MAX_TOKENS \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --n_eval_per_dataset 50 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --eval_steps 1 \
    --output_dir $OUTPUT_DIR/sft_train
