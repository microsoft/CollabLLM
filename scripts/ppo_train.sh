#!/bin/bash

# Source shared parameters
source scripts/config.sh

# Ensure a dataset name is provided as input
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

# Set dataset-specific parameters, start with sft model
set_dataset_config "$1"
set_assistant_model "$1" "sft"

RANDOM_SEED=$$
PORT=$((56410 + RANDOM_SEED % 10))

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --master_port=$PORT --nnodes=1 --nproc_per_node=6 \
CUDA_VISIBLE_DEVICES=0,1,2,3,6,7 torchrun --master_port=$PORT --nnodes=1 --nproc_per_node=6 \
    scripts/ppo_train.py \
    --datasets org_name/collabllm-$DATASET \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --user_model_name $USER_MODEL \
    --reward_model claude-3-5-sonnet-20240620 \
    --gradient_accumulation_steps 4 \
    --mini_batch_size 2 \
    --batch_size 8 \
    --window_size 2 \
    --learning_rate 2e-6 \
    --output_dir $OUTPUT_DIR/ppo_train/ \
    --max_new_tokens $MAX_TOKENS \
    --max_query_tokens 4096 \
    --save_step 10 \
    --num_samples 2 \
    --num_train_epochs 5 \
    --ppo_epochs 2 \
    --max_workers 1 \
    --llm_rw_weight 1 \
    --task_weight 1 \
    --cost_weight $COST_WEIGHT \
    --n_eval_per_dataset 50 \
    --push_to_hub
