# chat_session_simulator.py
from __future__ import annotations

import os
import copy
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import random
import torch
from transformers import pipeline, PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from collabllm.prompts import SYSTEM_PROMPT
from collabllm import ENABLE_COLLABLLM_LOGGING
from collabllm.prompts import COLLABLLM_TERMINATION_SIGNAL
from collabllm.modules import LLMCollaborator, UserSimulator
from collabllm.utils.format import is_conversational

logger = logging.getLogger(__name__)

PEFT_CHECKPOINT_SUFFIX = "-peft-checkpoint"

# Valid VLLM sampling parameters
VALID_VLLM_SAMPLING_PARAMS = {
    "n", "best_of", "presence_penalty", "frequency_penalty", "repetition_penalty",
    "temperature", "top_p", "top_k", "min_p", "seed", "stop", "stop_token_ids",
    "bad_words", "ignore_eos", "max_tokens", "min_tokens", "logprobs",
    "prompt_logprobs", "detokenize", "skip_special_tokens",
    "spaces_between_special_tokens", "truncate_prompt_tokens"
}

class ChatSessionSimulator:
    """Manages multiple simultaneous chat sessions."""

    # --------------------------------------------------------------------------- #
    # ChatSessionSimulator.run_chat_simulation                                    #
    # --------------------------------------------------------------------------- #
    def run_chat_simulation(
        self,
        *,
        task_desc: str,
        single_turn_prompt: str,
        chat_history: List[Dict[str, str]],
        assistant_generation_kwargs: Dict[str, Any],
        user_generation_kwargs: Dict[str, Any],
        num_samples: int = 1,                      
        max_new_turns: int = 0,
        proact_prompt_ratio: int = 0.0,
        add_system_prompt_ratio: float = 0.0,
        local_model: Optional[PreTrainedModel] = None,
        local_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        vllm_base_model: Optional = None,
        max_workers: int = 8,
        verbose: bool = True,
    ) -> List[List[Dict[str, str]]]:
        """
        Simulate *num_samples* conversations in parallel (internally batched).

        Returns
        -------
        List[List[Dict[str, str]]]
            A list of `num_samples` full chat transcripts.
        """
        # ------------------------------------------------------------------ #
        # 0 · Validation / defaults                                          #
        # ------------------------------------------------------------------ 
        self._validate_session_inputs(
            task_desc,
            single_turn_prompt,
            max_new_turns,
            local_model,
            local_tokenizer,
            vllm_base_model,
            assistant_generation_kwargs,
            user_generation_kwargs
        )

        # ------------------------------------------------------------------ #
        # 1 · Per-conversation state                                         #
        # ------------------------------------------------------------------ #
        sessions: List[List[Dict[str, str]]] = [
            copy.deepcopy(chat_history or []) for _ in range(num_samples)
        ]
        # for increasing sample diversity
        for sess in sessions[:int(num_samples * add_system_prompt_ratio)]:
            sess.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        current_roles = [
            self._determine_starting_role(hist) for hist in sessions
        ]
        active: set[int] = set(range(num_samples))          # indices still alive

        user_sims = [
            UserSimulator(
                task_desc=task_desc,
                single_turn_prompt=single_turn_prompt,
                **user_generation_kwargs,
            )
            for _ in range(num_samples)
        ]

        # Optional PEFT materialisation for vLLM
        model_name = assistant_generation_kwargs.get("model")
        if (
            vllm_base_model is not None
            and local_model is not None
            and hasattr(local_model, "peft_config")
        ):
            self._write_peft_checkpoint(local_model, model_name)

        # ------------------------------------------------------------------ #
        # 2 · Conversation loop (respects max_new_turns budget)              #
        # ------------------------------------------------------------------ #
        msg_budget = [max_new_turns for _ in range(num_samples)]  # ← NEW
        active: set[int] = {i for i, b in enumerate(msg_budget) if b > 0}

        pbar = tqdm(total=max_new_turns, desc="Simulating chat", disable=not (ENABLE_COLLABLLM_LOGGING and verbose))

        while active:
            # ---------- USER TURNS ---------------------------------------- #
            user_idx = [i for i in active if current_roles[i] == "user"]
            if user_idx:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    fut_to_i = {pool.submit(user_sims[i], sessions[i]): i for i in user_idx}
                    for fut in as_completed(fut_to_i):
                        i = fut_to_i[fut]
                        resp = fut.result()
                        self._log_response(f"user (Turn {len(sessions[i])})", resp)
                        sessions[i].append({"role": "user", "content": resp})
                        
                        msg_budget[i] -= 1

                        # early exit checks
                        if msg_budget[i] == 0 or self._should_terminate_conversation(resp):
                            current_roles[i] = "terminated"
                            active.discard(i)
                        else:
                            current_roles[i] = "assistant"
                    pbar.update(1)

            if not active:  # all dialogues exhausted their budget / terminated
                break

            # ---------- ASSISTANT TURNS ----------------------------------- #
            asst_idx = [i for i in active if current_roles[i] == "assistant"]
            if not asst_idx:
                continue

                
            # --- generate assistant replies (batched or threaded) --- #
            if local_model is None and vllm_base_model is None:
                num_asst = len(asst_idx)
                cutoff = int(num_asst * proact_prompt_ratio)

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    fut_to_i = {}
                    for rank, i in enumerate(asst_idx):
                        method_i = "proact" if rank < cutoff else "none"
                        collab_i = LLMCollaborator(method=method_i, **assistant_generation_kwargs)
                        fut = pool.submit(collab_i, sessions[i])
                        fut_to_i[fut] = i

                    responses = {fut_to_i[f]: f.result() for f in fut_to_i}
            else:
                batch_sess = [sessions[i] for i in asst_idx]
                if vllm_base_model is not None:
                    outs = self._batch_generate_with_vllm(
                        batch_sess,
                        vllm_base_model,
                        local_model,
                        model_name,
                        assistant_generation_kwargs,
                    )
                else:
                    outs = self._batch_generate_with_huggingface(
                        batch_sess,
                        local_model,
                        local_tokenizer,
                        assistant_generation_kwargs,
                    )
                responses = {g: r for g, r in zip(asst_idx, outs)}

            # --- post-process assistant replies --- #
            for i, resp in responses.items():
                self._log_response(f"assistant (Turn {len(sessions[i])})", resp)
                sessions[i].append({"role": "assistant", "content": resp})

                msg_budget[i] -= 1

                if msg_budget[i] == 0:
                    current_roles[i] = "terminated"
                    active.discard(i)
                else:
                    current_roles[i] = "user"
            pbar.update(1)

        pbar.close()
        return sessions

    # ------------------------------------------------------------------ #
    # Batch generators                                                   #
    # ------------------------------------------------------------------ #
    def _batch_generate_with_vllm(
        self,
        batch_messages: List[List[Dict[str, str]]],
        vllm_base_model,
        local_model,
        model_name: str,
        generation_kwargs: Dict[str, Any],
        return_outputs: bool = False
    ) -> List[str]:
        """Batched vLLM generation (returns list of responses)."""
        from vllm.lora.request import LoRARequest

        sampling_params = self._convert_to_sampling_params(generation_kwargs)
        peft_dir = self._get_peft_dir(model_name)
        if hasattr(local_model, "peft_config") and os.path.exists(peft_dir):
            logger.info(f"Using PEFT checkpoint from {peft_dir}")
            lora_req = LoRARequest("interactive_adapter", 1, peft_dir)
        else:
            lora_req = None

        # vLLM can accept a list of message histories; returns list[str]
        if is_conversational({"prompt": batch_messages[0]}):
            outs = vllm_base_model.chat(batch_messages, sampling_params=sampling_params, lora_request=lora_req, use_tqdm=False)
        else:
            outs = vllm_base_model.generate(batch_messages, sampling_params=sampling_params, lora_request=lora_req, use_tqdm=False)

        if return_outputs:
            return outs
        
        generated_texts = [out.outputs[0].text for out in outs]
        return generated_texts

    def _batch_generate_with_huggingface(
        self,
        batch_messages: List[List[Dict[str, str]]],
        local_model,
        local_tokenizer,
        generation_kwargs: Dict[str, Any],
    ) -> List[str]:
        """Batched HF generation (one forward pass)."""
        torch.cuda.empty_cache()
        local_tokenizer.padding_side = "left"
        local_tokenizer.pad_token = local_tokenizer.eos_token

        generator = pipeline(
            "text-generation",
            model=local_model,
            tokenizer=local_tokenizer,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )

        generation_kwargs = copy.deepcopy(generation_kwargs)
        max_new = generation_kwargs.pop("max_tokens", 1024)
        generation_kwargs.pop("model", None)  # not needed for HF pipeline
        prompts = [msgs for msgs in batch_messages]  # HF pipeline accepts list
        outputs = generator(
            prompts,
            max_new_tokens=max_new,
            **generation_kwargs,
        )

        # Extract only the newly generated part for each item
        results = []
        for prompt_msgs, out in zip(prompts, outputs):
            if isinstance(out, list):
                out = out[0]  # HF pipeline returns list of dicts
            full_text = out["generated_text"]

            if isinstance(prompt_msgs, str):
                results.append(full_text[len(prompt_msgs) :])
            else:
                results.append(full_text[-1]["content"])
        torch.cuda.empty_cache()
        return results

    # ------------------------------------------------------------------ #
    # … the rest of the helper methods (_validate_session_inputs, etc.)  #
    # ------------------------------------------------------------------ #
    def _write_peft_checkpoint(
        self,
        local_model,
        model_name: str
    ):
        """
        Write the PEFT checkpoint for the local model if required.
        
        Args:
            local_model: The local model instance to save
            model_name: Name of the assistant model
            
        Returns:
            Path to the saved PEFT checkpoint directory
            
        Raises:
            FileNotFoundError: If unable to create or access the run user directory
        """

        peft_dir = self._get_peft_dir(model_name)
        
        # Ensure PEFT directory exists
        os.makedirs(peft_dir, exist_ok=True)
        
        # Save local model as PEFT checkpoint
        local_model.save_pretrained(peft_dir)
        logger.debug(f"Saved PEFT checkpoint to {peft_dir}")
        
        return peft_dir
    
    def _validate_session_inputs(
        self,
        task_desc: str,
        single_turn_prompt: str,
        max_new_turns: int,
        local_model: Optional[PreTrainedModel] = None,
        local_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        vllm_base_model: Optional[str] = None,
        assistant_generation_kwargs: Optional[Dict[str, Any]] = None,
        user_generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Sanity-check all arguments before starting a chat session.

        Raises
        ------
        ValueError
            If any invariant required by the session runner is violated.
        """
        if not isinstance(task_desc, str) or not task_desc.strip():
            raise ValueError("`task_desc` must be a non-empty string.")

        if not isinstance(single_turn_prompt, str) or not single_turn_prompt.strip():
            raise ValueError("`single_turn_prompt` must be a non-empty string.")

        if not isinstance(max_new_turns, int) or max_new_turns < 0:
            raise ValueError("`max_new_turns` must be an integer ≥ 0.")

        if (local_model is None) ^ (local_tokenizer is None):
            raise ValueError(
                "Provide *both* `local_model` and `local_tokenizer`, or neither."
            )

        if assistant_generation_kwargs.get("model") is None:
            raise ValueError(
                "`assistant_generation_kwargs` must include a 'model' key."
            )
        if user_generation_kwargs.get("model") is None:
            raise ValueError(
                "`user_generation_kwargs` must include a 'model' key."
            )

    def _determine_starting_role(self, chat_history: List[Dict[str, str]]) -> str:
        """Determine which role should start the conversation."""
        if chat_history and chat_history[-1]['role'] == 'user':
            return 'assistant'
        return 'user'
    
    def _get_peft_dir(self, model_name: str) -> str:
        """
        Get the directory for the PEFT checkpoint.
        
        Args:
            model_name: Name of the assistant model
            
        Returns:
            Path to the PEFT checkpoint directory
        """
        run_user_dir = os.environ.get("RUN_USER_DIR")
        return os.path.join(run_user_dir, f"{model_name.replace('/', '_')}{PEFT_CHECKPOINT_SUFFIX}")
    
    def _convert_to_sampling_params(self, generation_kwargs: Dict[str, Any]):
        """
        Convert generation kwargs to VLLM SamplingParams.
        
        Args:
            generation_kwargs: Dictionary of generation parameters
            
        Returns:
            SamplingParams instance for VLLM
        """
        from vllm.sampling_params import SamplingParams

        # Filter valid parameters
        generation_kwargs = copy.deepcopy(generation_kwargs)
        generation_kwargs.pop('model', None)  # 'model' is not a sampling param
        sampling_kwargs = {'max_tokens': 1024}  # Default max_tokens
        unmapped_params = []
        
        for key, value in generation_kwargs.items():
            if key in VALID_VLLM_SAMPLING_PARAMS:
                sampling_kwargs[key] = value
            else:
                unmapped_params.append(key)
        
        # Log unmapped parameters
        if unmapped_params:
            logger.warning(f"Unmapped VLLM parameters: {unmapped_params}")
        
        return SamplingParams(**sampling_kwargs)
    
    def _should_terminate_conversation(self, response: str) -> bool:
        """
        Check if the response contains a termination signal.
        
        Args:
            response: The response text to check
            
        Returns:
            True if conversation should terminate, False otherwise
        """
        try:
            return COLLABLLM_TERMINATION_SIGNAL in response
        except Exception as e:
            logger.error(f"Error checking for chat termination: {e}")
            return False
    
    def _log_response(self, role: str, response: str) -> None:
        """Log the response if verbose mode is enabled and on main process."""
        logger.info(f"[rank {os.environ.get('RANK', 0)}]{role.capitalize()}: {response}")


