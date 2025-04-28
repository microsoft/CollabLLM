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

# Set dataset-specific parameters, start with trained offline model
set_dataset_config "$DATASET"
set_assistant_model "$DATASET" "dpo_offline"

RANDOM_SEED=$$
PORT=$((56420 + RANDOM_SEED % 10))
DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --master_port=$PORT --nnodes=1 --nproc_per_node=$NUM_GPUS \
    scripts/dpo_train_online.py \
    --dataset org_name/collabllm-$DATASET \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --user_model_name gpt-4o-mini \
    --reward_model claude-3-5-sonnet-20240620 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size $DPO_BATCH_SIZE \
    --num_train_epochs 1 \
    --save_total_limit 20 \
    --max_new_tokens $MAX_TOKENS \
    --minimum_gap $GAP \
    --temperature $TEMP \
    --cost_weight $COST_WEIGHT \
    --max_query_tokens 4096 \
    --learning_rate 5e-6 \
    --window_size 2 \
    --top_p 0.9 \
    --num_samples 3 \
    --llm_rw_weight 1 \
    --task_weight 1 \
    --max_workers 1 \
    --output_dir $OUTPUT_DIR/dpo_train_online
