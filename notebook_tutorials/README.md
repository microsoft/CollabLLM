# CollabLLM Tutorials

This directory contains lightweight tutorials for using CollabLLM with your own tasks and models. These notebooks demonstrate how to construct datasets and compute multiturn-aware rewards (MR) for both API-based and local LLMs.

---

## Notebooks Overview

### 📘 `build_datasets.ipynb`
Learn how to create and register:
- `SingleTurnDataset`: Define your own task-specific dataset.
- `MultiturnDataset`: Structure multiturn conversations for training and evaluation (e.g., SFT, RL).

### ⚙️ `compute_mr_api_llm.ipynb`
Compute Multiturn-aware Rewards using API-accessible LLMs (e.g., OpenAI, Anthropic) via:
```python
from collabllm.reward import multiturn_aware_reward
```

### ⚙️ `compute_mr_local_llm.ipynb`
Same as above, but for local LLMs (e.g., LLaMA or Mistral) loaded through transformers.
 
