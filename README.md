# CollabLLM: From Passive Responders to Active Collaborators

<div align="left">

[![](https://img.shields.io/badge/Website-CollabLLM-purple?style=plastic&logo=Google%20Chrome)](http://aka.ms/CollabLLM)
[![](https://img.shields.io/badge/Datasets_&_Models-HuggingFace-yellow?style=plastic&logo=Hugging%20Face)](https://huggingface.co/collabllm)
[![](https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arxiv)](https://arxiv.org/pdf/2502.00640)
[![](https://img.shields.io/badge/PyPI-collabllm-brightgreen?style=plastic&logo=Python)](https://pypi.org/project/collabllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

📢 Oral @ ICML 2025 (1% out of all submissions)

## Overview

CollabLLM transforms traditional language models from passive responders to active collaborators in multi-turn conversations. This repository provides the complete framework for computing multiturn-aware rewards and training collaborative language models.

**Note**: The code on this page is no longer maintained. For the latest updates, please visit the [repository](https://github.com/Wuyxin/collabllm) maintained by the first author.

---
## Installation

To get started, create a new environment and install `collabllm` via [pip](https://pypi.org/project/collabllm/):

```bash
conda create -n collabllm python=3.10
conda activate collabllm
pip install collabllm
```

### Optional: For distributed training
If you need distributed training:

```bash
conda install deepspeed mpi4py -c conda-forge
```

### Optional: For customized datasets and metrics
You may install additional packages (e.g., `pip install bigcodebench matplotlib`) for task-specific metrics or evaluation.

## Quick Start

- Lightweight usage: Compute Multiturn-aware Rewards (MRs) for any model responses and construct datasets following `notebook_tutorials/`.
- Synthetic data generation: Generating high-quality synthetic conversational data following `scripts/engine/build_dataset.py`.
- Train CollabLLM: Conduct SFT/DPO/PPO models training to maximize MRs following examples under `scripts/train/*.py`.


### Add Your Own Task

To apply CollabLLM to a new task:

1. **Add a Dataset:**
   Place your single-turn dataset in `examples/single_turn_ds/` and register it in `__init__.py`.

2. **(Optional) Add Metrics:**
   Add new metrics to `examples/metrics/` and register them in `__init__.py`.

You can now run data generation, reward computation, and model training using your customized setup.


## Citation

If you find our work useful in your research, please cite the following:

```bibtex
@inproceedings{collabllm2025,
    title={CollabLLM: From Passive Responders to Active Collaborators},
    author={Shirley Wu and Michel Galley and Baolin Peng and Hao Cheng and
            Gavin Li and Yao Dou and Weixin Cai and James Zou and
            Jure Leskovec and Jianfeng Gao},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
