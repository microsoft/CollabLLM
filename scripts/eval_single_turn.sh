#!/bin/bash

# Usage: ./evaluate.sh <dataset>
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

# Parameters
DATASET_NAME=$1

# Load configurations
source scripts/config.sh
set_dataset_config $DATASET_NAME

# Random seed and port setup
RANDOM_SEED=$$
PORT=$((56440 + RANDOM_SEED % 10))

# Output directory
OUTDIR=$OUTPUT_DIR/eval/base_single_turn

# Run evaluation
CUDA_VISIBLE_DEVICES=7 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_single_turn.py \
    --dataset $DATASET \
    --output $OUTDIR \
    --split test \
    --judge_model claude-3-5-sonnet-20240620 \
    --assistant_model_name meta-llama/Llama-3.1-8B-Instruct \
    --temperature $TEMP \
    --n_eval $N_EVAL \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 
