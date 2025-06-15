import os
import os.path as osp

current_dir = osp.dirname(__file__)

COLLABLLM_TERMINATION_SIGNAL = os.getenv('COLLABLLM_TERMINATION_SIGNAL', "[[TERMINATE CHAT]]")
with open(osp.join(current_dir, 'system_prompt.txt'), 'r') as f:
    SYSTEM_PROMPT = f.read()

with open(osp.join(current_dir, 'proact_instruction.txt'), 'r') as f:
    PROACT_MODEL_PROMPT = f.read()

with open(osp.join(current_dir, 'extract_multiturn_completion.txt'), 'r') as f:
    EXTRACT_MULTITURN_COMPLETION_PROMPT = f.read()

with open(osp.join(current_dir, 'user_simulator.txt'), 'r') as f:
    USER_SIMULATOR_PROMPT = f.read()
