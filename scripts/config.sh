#!/bin/bash

# Common settings
export USE_SUB=false
export USE_SNAP=true
export USE_GCR=false

# Define the base output directory
OUTPUT_DIR=/name/project/collabllm/github/outputs

# Dataset configurations
function set_dataset_config() {
    case "$1" in
        math-hard)
            DATASET=math-hard
            TEMP=0.6
            N_EVAL=200
            MAX_NEW_TURNS=6
            MAX_TOKENS=1536
            USER_MODEL=gpt-4o-mini
            EVAL_USER_NAME=gpt-4o-mini
            GAP=0.05  # Gap for DPO training
            DPO_BATCH_SIZE=2
            COST_WEIGHT=5e-4
            ;;
        medium)
            DATASET=medium
            TEMP=0.8
            N_EVAL=100
            MAX_NEW_TURNS=8
            MAX_TOKENS=2048
            USER_MODEL=gpt-4o
            EVAL_USER_NAME=gpt-4o
            GAP=0.02  # Gap for DPO training
            DPO_BATCH_SIZE=1
            COST_WEIGHT=1e-4
            ;;
        bigcodebench)
            DATASET=bigcodebench
            TEMP=0.6
            N_EVAL=600
            MAX_NEW_TURNS=6
            MAX_TOKENS=1536
            USER_MODEL=gpt-4o-mini
            EVAL_USER_NAME=gpt-4o
            GAP=0.02  # Gap for DPO training
            DPO_BATCH_SIZE=1
            COST_WEIGHT=5e-4
            ;;
        *)
            echo "Unknown dataset: $1"
            exit 1
            ;;
    esac
}

# Dataset-specific base directories
MATH_HARD_DIR=collabllm-math-hard
MEDIUM_DIR=collabllm-medium
BIGCODEBENCH_DIR=collabllm-bigcodebench

# Define training mode directories
SFT_DIR=sft_train
DPO_ONLINE_DIR=dpo_train_online
DPO_OFFLINE_DIR=dpo_train_offline
PPO_DIR=ppo_train


# Assistant model configurations
ASSISTANT_MODELS=(
    # Format: "dataset-mode trained_model_path prompt_method add_system_prompt"

    # Math-hard configurations
    "math-hard-base meta-llama/Llama-3.1-8B-Instruct none False"
    "math-hard-base_collabllm meta-llama/Llama-3.1-8B-Instruct collabllm True"
    "math-hard-sft $OUTPUT_DIR/$SFT_DIR/$MATH_HARD_DIR/Llama-3.1-8B-Instruct_epoch-3_lr-2e-05/checkpoint-4 none True"
    "math-hard-dpo_online $OUTPUT_DIR/$DPO_ONLINE_DIR/$MATH_HARD_DIR/dpo_dpo_Llama-3.1-8B-Instruct_epoch-3_lr-2e-05_checkpoint-4_epoch-8_lr-5e-06_gap-0.05_2024-11-10-08-05_epoch-1_lr-5e-06_gap-0.05_2024-11-21-08-51/checkpoint-10 none True"
    "math-hard-dpo_offline $OUTPUT_DIR/$DPO_OFFLINE_DIR/$MATH_HARD_DIR/dpo_Llama-3.1-8B-Instruct_epoch-3_lr-2e-05_checkpoint-4_epoch-8_lr-5e-06_gap-0.05 none True"
    "math-hard-ppo $OUTPUT_DIR/$PPO_DIR/$MATH_HARD_DIR/ppo_Llama-3.1-8B-Instruct_epoch-3_lr-1e-05/step_60 none True"

    # Medium configurations
    "medium-base meta-llama/Llama-3.1-8B-Instruct none False"
    "medium-base_collabllm meta-llama/Llama-3.1-8B-Instruct collabllm True"
    "medium-sft $OUTPUT_DIR/$SFT_DIR/$MEDIUM_DIR/Llama-3.1-8B-Instruct_epoch-3_lr-1e-05 none True"
    "medium-dpo_online $OUTPUT_DIR/$DPO_ONLINE_DIR/medium/online_dpo_dpo_Llama-3.1-8B-Instruct_epoch-3_lr-1e-05_epoch-8_lr-5e-06_gap-0.02_2024-11-27-04-31_epoch-1_lr-5e-06_gap-0.02_2024-11-30-11-32/checkpoint-5 none True"
    "medium-dpo_offline $OUTPUT_DIR/$DPO_OFFLINE_DIR/$MEDIUM_DIR/dpo_Llama-3.1-8B-Instruct_epoch-3_lr-1e-05_epoch-8_lr-5e-06_gap-0.02_2024-11-27-04-31 none True"
    "medium-ppo $OUTPUT_DIR/$PPO_DIR/$MEDIUM_DIR/ppo_Llama-3.1-8B-Instruct_epoch-3_lr-1e-05_2024-11-27-11-24 none True"

    # Bigcodebench configurations
    "bigcodebench-base meta-llama/Llama-3.1-8B-Instruct none False"
    "bigcodebench-base_collabllm meta-llama/Llama-3.1-8B-Instruct collabllm True"
    "bigcodebench-sft $OUTPUT_DIR/$SFT_DIR/$BIGCODEBENCH_DIR/Llama-3.1-8B-Instruct_epoch-3_lr-1e-05 none True"
    "bigcodebench-dpo_offline $OUTPUT_DIR/$DPO_OFFLINE_DIR/$BIGCODEBENCH_DIR/dpo_Llama-3.1-8B-Instruct_epoch-3_lr-1e-05_epoch-8_lr-5e-06_gap-0.02_2024-12-01-09-38/checkpoint-37 none True"
    "bigcodebench-dpo_online $OUTPUT_DIR/$DPO_ONLINE_DIR/$BIGCODEBENCH_DIR/online_dpo_dpo_Llama-3.1-8B-Instruct_epoch-3_lr-1e-05_epoch-8_lr-5e-06_gap-0.02_2024-12-01-09-38_checkpoint-37_epoch-1_lr-5e-06_gap-0.02_2024-12-23-08-45/checkpoint-9 none True"
    "bigcodebench-ppo $OUTPUT_DIR/$PPO_DIR/$BIGCODEBENCH_DIR/ppo_Llama-3.1-8B-Instruct_epoch-3_lr-1e-05_2024-12-18-02-19/step_40 none True"
)

# Assistant model configurations
function set_assistant_model() {
    for config in "${ASSISTANT_MODELS[@]}"; do
        IFS=" " read -r key model_path prompt_method add_system_prompt <<< "$config"
        if [ "$1-$2" == "$key" ]; then
            ASSISTANT_MODEL_NAME="$model_path"
            PROMPT_METHOD="$prompt_method"
            ADD_SYSTEM_PROMPT="$add_system_prompt"
            OUTPUT_DIR_SUFFIX="${2}" # Use mode as the suffix for the output directory
            return
        fi
    done
    echo "Unknown dataset-mode pair: $1-$2"
    exit 1
}
