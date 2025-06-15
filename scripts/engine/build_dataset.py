"""
To run the following, you need:
    - A dataset implemented under `examples/single_turn_ds`
    - (Optional) Any custom metrics implemented under `examples/metrics`

Example Usage:
On math-hard:
-------
    python -m scripts.engine.build_dataset \
        --dataset_name math-hard \
        --metric_names "accuracy" "interactivity" "token_amount" \
        --metric_weights 1 0.5 -0.5 \
        --user_generation_kwargs '{"model": "gpt-4o"}' \
        --assistant_generation_kwargs '{"model": "gpt-4o", "temperature": 0.8}' \
        --reward_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
        --output_dir outputs/multiturn_data \
        --train_size 100 \
        --num_candidate_responses 3 \
        --hf_entity collabllm

On medium:
-------
    python -m scripts.engine.build_dataset \
        --dataset_name medium \
        --metric_names "document->bleu" "token_amount" \
        --metric_weights 1 -0.1 \
        --user_generation_kwargs '{"model": "gpt-4o"}' \
        --assistant_generation_kwargs '{"model": "gpt-4o", "temperature": 0.9}' \
        --reward_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
        --output_dir outputs/multiturn_data \
        --train_size 200 \
        --num_candidate_responses 3 \
        --hf_entity collabllm

On bigcodebench:
-------
    python -m scripts.engine.build_dataset \
        --dataset_name bigcodebench \
        --metric_names "runnable code->pass_rate" "token_amount" \
        --metric_weights 1 -0.5 \
        --user_generation_kwargs '{"model": "gpt-4o"}' \
        --assistant_generation_kwargs '{"model": "gpt-4o", "temperature": 0.8}' \
        --reward_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
        --output_dir outputs/multiturn_data \
        --train_size 200 \
        --num_candidate_responses 3 \
        --hf_entity collabllm
"""

import argparse
import hashlib
import json
import os
import os.path as osp
from tqdm import tqdm
from typing import List, Dict, Any
from dotenv import load_dotenv
import warnings
import concurrent.futures

from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.synthetic import generate_multiturn_dataset
from examples.single_turn_ds import datasets_info
from examples.metrics import *


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def data_engine(args):
    dataset_cls = datasets_info[args.dataset_name]["class"]
    task_desc = datasets_info[args.dataset_name]["task_desc"]
    dataset = dataset_cls().to_hf_dataset()
    train = (
        dataset["train"].select(range(args.train_size))
        if args.train_size > 0
        else dataset["train"]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = osp.join(args.output_dir, f"{args.dataset_name}_multiturn.json")

    data_list: List[Dict[str, Any]] = []
    seen_prompt_hashes = set()

    if osp.exists(output_path):
        if args.resume:
            with open(output_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
            seen_prompt_hashes = {
                compute_hash(ex["single_turn_prompt"]) for ex in data_list
            }
        else:
            warnings.warn(
                "Output file already exists. Use --resume to continue from the last saved state.",
                UserWarning,
            )
            return

    # Filter out examples whose prompt‐hash is already in seen_prompt_hashes
    pending_examples = [
        ex
        for ex in train
        if compute_hash(ex["single_turn_prompt"]) not in seen_prompt_hashes
    ]

    if not pending_examples:
        print("No new examples to generate (all seen).")
        return

    # Create a ThreadPoolExecutor with max_gen_workers threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_gen_workers) as executor:
        future_to_hash = {}
        for example in pending_examples:
            prompt_hash = compute_hash(example["single_turn_prompt"])

            # Submit generate_multiturn_dataset using kwargs
            future = executor.submit(
                generate_multiturn_dataset,
                task_desc=task_desc,
                single_turn_prompt=example["single_turn_prompt"],
                single_turn_completion=example["single_turn_completion"],
                single_turn_metadata=example["single_turn_metadata"],
                metric_names=args.metric_names,
                user_generation_kwargs=args.user_generation_kwargs,
                assistant_generation_kwargs=args.assistant_generation_kwargs,
                reward_generation_kwargs=args.reward_generation_kwargs,
                metric_weights=args.metric_weights,
                proact_prompt_ratio=args.proact_prompt_ratio,
                num_candidate_responses=args.num_candidate_responses,
                max_total_turns=args.max_total_turns,
                max_new_turns=args.max_new_turns,
                num_samples=args.num_samples,
                max_workers=min(args.num_samples, 4),
                max_metric_workers=args.max_metric_workers,
                add_system_prompt_ratio=args.add_system_prompt_ratio,
            )

            future_to_hash[future] = prompt_hash

        # Use tqdm to show progress as each future completes
        for future in tqdm(
            concurrent.futures.as_completed(future_to_hash),
            total=len(future_to_hash),
            desc="Generating multi-turn conversations",
        ):
            prompt_hash = future_to_hash[future]
            try:
                multiturn_data = future.result()
            except Exception as e:
                print(f"Error generating for prompt {prompt_hash}: {e}")
                continue

            if multiturn_data is None:
                continue
            
            data_list.append(multiturn_data)
            seen_prompt_hashes.add(prompt_hash)

            # Write to JSON after each new conversation
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_list, f, indent=2)

            # Push to Hugging Face Hub incrementally
            MultiturnDataset(data_list).push_to_hub(
                repo_id=f"{args.hf_entity}/collabllm-multiturn-{args.dataset_name}"
            )


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-turn synthetic conversations with metrics.")

    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the single-turn dataset.")
    parser.add_argument("--metric_names", nargs="+", required=True, help="List of evaluation metric names.")
    parser.add_argument("--user_generation_kwargs", type=json.loads, default="{}", help="JSON dict of generation kwargs for user.")
    parser.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}", help="JSON dict of generation kwargs for assistant.")
    parser.add_argument("--reward_generation_kwargs", type=json.loads, default="{}", help="Optional JSON dict for reward generation.")
    parser.add_argument("--metric_weights", type=float, nargs="+", default=None, help="Optional weights for each metric.")
    parser.add_argument("--proact_prompt_ratio", type=float, default=0.5, help="0 for none, 1 for proact, 0~1 for mixed.")
    parser.add_argument("--add_system_prompt_ratio", type=float, default=0, help="0 for none, 1 for proact, 0~1 for mixed.")
    parser.add_argument("--num_candidate_responses", type=int, default=2, help="Number of assistant candidates per turn.")
    parser.add_argument("--max_total_turns", type=int, default=14, help="Maximum number of conversation turns.")
    parser.add_argument("--max_new_turns", type=int, default=4, help="Window size for context in multi-turn generation.")
    parser.add_argument("--num_samples", type=int, default=3, help="Sample size for generating multiple conversations in one batch.")
    parser.add_argument("--train_size", type=int, default=500, help="Number of conversations to generate.")
    parser.add_argument("--max_workers", type=int, default=16, help="Maximum number of parallel workers for sampling conversations.")
    parser.add_argument("--max_metric_workers", type=int, default=16, help="Maximum number of parallel workers for metrics.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated output.")
    parser.add_argument("--hf_entity", type=str, required=True, help="Hugging Face user or organization for dataset upload.")
    parser.add_argument("--save_steps", type=int, default=10, help="Save intermediate results every N steps.")
    parser.add_argument("--resume", action="store_true", help="Resume from the last saved state if available.")
    parser.add_argument("--max_gen_workers", type=int, default=8, help="Maximum number of threads to use for generating conversations (ThreadPool size).")

    load_dotenv(".env")

    args = parser.parse_args()

    print(args)
    data_engine(args)
