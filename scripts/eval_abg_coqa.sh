#!/bin/bash

# Usage: ./evaluate.sh

# Load configurations
source scripts/config.sh
set_dataset_config bigcodebench
set_assistant_model bigcodebench dpo_online

# Random seed and port setup
RANDOM_SEED=$$
PORT=$((56440 + RANDOM_SEED % 10))

# Output directory
OUTDIR=$OUTPUT_DIR/eval/abg_coqa

# CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
#     --nnodes=1 --nproc_per_node=1 \
#     scripts/eval_single_turn.py \
#     --dataset abg-coqa \
#     --output $OUTDIR \
#     --split test \
#     --judge_model claude-3-5-sonnet-20240620 \
#     --assistant_model_name meta-llama/Llama-3.1-8B-Instruct \
#     --temperature $TEMP \
#     --n_eval 2000 \
#     --max_new_tokens $MAX_TOKENS \
#     --top_p 0.9 


# CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$PORT \
#     --nnodes=1 --nproc_per_node=1 \
#     scripts/eval_single_turn.py \
#     --dataset abg-coqa \
#     --output $OUTDIR \
#     --split test \
#     --judge_model claude-3-5-sonnet-20240620 \
#     --assistant_model_name gpt-4o  \
#     --temperature $TEMP \
#     --n_eval 2000 \
#     --max_new_tokens $MAX_TOKENS \
#     --top_p 0.9 


# # Run evaluation
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_single_turn.py \
    --dataset abg-coqa \
    --output $OUTDIR \
    --split test \
    --judge_model claude-3-5-sonnet-20240620 \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --temperature $TEMP \
    --n_eval 2000 \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 \
    --add_sys_prompt
