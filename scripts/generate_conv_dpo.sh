export USE_SUB=false
export USE_SNAP=true
export USE_GCR=false

# DATASET=math-hard
# TEMP=0.6
# MAX_NEW_TURNS=6
# MAX_TOKENS=1536
# COST_WEIGHT=5e-4
# LLM_RW_WEIGHT=1
# USER_MODEL=gpt-4o-mini
# N_EVAL=200

# DATASET=medium
# TEMP=0.8
# MAX_NEW_TURNS=8
# MAX_TOKENS=2048
# COST_WEIGHT=1e-4
# LLM_RW_WEIGHT=0
# USER_MODEL=gpt-4o
# N_EVAL=100

DATASET=bigcodebench
TEMP=0.5
MAX_NEW_TURNS=8
MAX_TOKENS=1536
COST_WEIGHT=5e-4
LLM_RW_WEIGHT=1
USER_MODEL=gpt-4o-mini
N_EVAL=600

# DATASET=humaneval
# TEMP=0.2
# MAX_NEW_TURNS=6
# COST_WEIGHT=1e-4

CUDA_VISIBLE_DEVICES=0 python scripts/generate_conv_dpo.py \
    --dataset $DATASET \
    --max_workers 10 \
    --num_samples 3 \
    --user_model_name $USER_MODEL \
    --assistant_model_name gpt-4o \
    --reward_model claude-3-5-sonnet-20240620 \
    --max_new_tokens $MAX_TOKENS \
    --max_new_turns $MAX_NEW_TURNS \
    --window_size 2 \
    --temperature $TEMP \
    --top_p 0.9 \
    --task_weight 1 \
    --llm_rw_weight $LLM_RW_WEIGHT \
    --cost_weight $COST_WEIGHT \
    --n_eval_per_dataset $N_EVAL \
    --max_num_conv 500 \
    --resume
