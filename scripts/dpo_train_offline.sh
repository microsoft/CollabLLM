#!/bin/bash

# Source shared parameters
source scripts/config.sh

# Ensure a dataset name and number of GPUs is provided as input
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset> <num_gpus>"
    exit 1
fi

DATASET="$1"
NUM_GPUS="$2"

# Set dataset-specific parameters, start with trained sft model
set_dataset_config "$DATASET"
set_assistant_model "$DATASET" "sft"

RANDOM_SEED=$$
PORT=$((56430 + RANDOM_SEED % 10))
DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))

CUDA_VISIBLE_DEVICES=$DEVICES WANDB__SERVICE_WAIT=300 torchrun --master_port=$PORT --nnodes=1 --nproc_per_node=$NUM_GPUS \
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
