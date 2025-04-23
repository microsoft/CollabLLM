#!/bin/bash

# Source shared parameters
source scripts/config.sh

# Ensure a dataset name is provided as input
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

# Set dataset-specific parameters, start with trained sft model
set_dataset_config "$1"
set_assistant_model "$1" "sft"

RANDOM_SEED=$$
PORT=$((56430 + RANDOM_SEED % 10))

CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 WANDB__SERVICE_WAIT=300 torchrun --master_port=$PORT --nnodes=1 --nproc_per_node=7 \
    scripts/dpo_train_offline.py \
    --datasets org_name/collabllm-$DATASET \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 2 \
    --num_train_epochs 8 \
    --save_total_limit 10 \
    --eval_steps 1 \
    --learning_rate 5e-6 \
    --n_eval_per_dataset 50 \
    --max_new_tokens $MAX_TOKENS \
    --minimum_gap $GAP \
    --output_dir $OUTPUT_DIR/dpo_train_offline
