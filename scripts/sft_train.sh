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

# Set dataset-specific parameters, start with base model
set_dataset_config "$DATASET"
set_assistant_model "$DATASET" "base"

RANDOM_SEED=$$
PORT=$((56400 + RANDOM_SEED % 10))
DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export TRITON_CACHE_DIR=$OUTPUT_DIR/cache

CUDA_VISIBLE_DEVICES=$DEVICES WANDB__SERVICE_WAIT=300 torchrun --master_port=$PORT --nnodes=1 --nproc_per_node=$NUM_GPUS \
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
