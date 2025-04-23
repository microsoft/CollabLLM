#!/bin/bash

# Usage: ./scripts/eval_multiturn.sh  <dataset> <mode>
# Usage: ./scripts/eval_multiturn.sh bigcodebench dpo_offline
# dpo_online dpo_offline
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset> <mode>"
    exit 1
fi

# Parameters
DATASET_NAME=$1
MODE=$2

# Load configurations
source scripts/config.sh
set_dataset_config $DATASET_NAME
set_assistant_model $DATASET_NAME $MODE

# Decide whether to add --add_sys_prompt
ADD_SYS_PROMPT_FLAG=""
if [ "$ADD_SYSTEM_PROMPT" == "True" ]; then
    ADD_SYS_PROMPT_FLAG="--add_sys_prompt"
fi

# Random seed and port setup
RANDOM_SEED=$$
PORT=$((56480 + RANDOM_SEED % 10))

# fix user model to gpt-4o for eval
USER_MODEL=gpt-4o

# Output directory
OUTDIR=$OUTPUT_DIR/eval/$OUTPUT_DIR_SUFFIX
# Run evaluation
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_multiturn.py \
    --dataset $DATASET \
    --output $OUTDIR \
    --split test \
    --judge_model claude-3-5-sonnet-20240620 \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --user_model_name $EVAL_USER_NAME \
    --prompt_method $PROMPT_METHOD \
    --temperature $TEMP \
    --max_new_turns $MAX_NEW_TURNS \
    --n_eval $N_EVAL \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 \
    $ADD_SYS_PROMPT_FLAG
