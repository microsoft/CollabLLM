---
# Machine-readable fields for downstream agents / dashboards.
stage: early-research
research-area: Models & Foundations
last-updated: 2026-05-12
---

# CollabLLM: From Passive Responders to Active Collaborators

**Team:** Microsoft Research — Deep Learning Group
**Contributors:** Shirley Wu, Michel Galley, Baolin Peng, Hao Cheng, Gagan Bansal, Saleema Amershi, Jure Leskovec, Jianfeng Gao, Swadheen Shukla

---

## What it is

CollabLLM is a training framework that teaches LLMs to collaborate with users across multi-turn conversations — knowing when to ask clarifying questions, when to commit to an answer, and how to adapt tone and communication style so the *whole* exchange succeeds, not just the immediate reply. Rather than optimizing each turn in isolation, CollabLLM places a model in a simulated dialogue with a user simulator, forward-samples how the conversation would play out, and uses those rollouts to score and update the model with reinforcement learning.

🔗 [Project page](https://aka.ms/CollabLLM) · [Paper (arXiv 2502.00640)](https://arxiv.org/abs/2502.00640) · [MSR blog](https://www.microsoft.com/en-us/research/blog/collabllm-teaching-llms-to-collaborate-with-users/)

## Core idea

**Claim:** the value of a model response isn't its single-turn helpfulness — it's the response's *causal contribution* to the conversation's eventual success, and training has to measure exactly that. Standard RLHF and DPO pipelines optimize next-turn preference, which rewards confident-sounding answers and quietly penalizes the clarifying question that would have unblocked the user three turns later. CollabLLM's insight is **Multiturn-aware Rewards (MR)**: for each candidate response, forward-sample several plausible continuations of the conversation with a user simulator, score each continuation along task success, efficiency (token cost), and interactivity (LLM-as-judge), and average those downstream scores back onto the original response. The model is then fine-tuned with that long-horizon signal using PPO, DPO (offline), or Online DPO on top of LoRA-adapted Llama-3.1-8B-Instruct. This is hard to replicate because it requires co-designing four pieces — a faithful user simulator, a forward-sampling window short enough to be tractable but long enough to be causal, intrinsic/extrinsic reward composition that doesn't collapse into reward hacking, and an RL recipe stable under noisy long-horizon credit assignment — and getting any one of them wrong reverts the system to single-turn behavior.

## Why it matters

**To the field:** Reframes alignment as a *multi-turn* problem and introduces a concrete, reusable mechanism (forward-sampled rewards over simulated dialogues) that other groups can apply to any conversational task. The paper received an [ICML 2025 Outstanding Paper Award](https://icml.cc/virtual/2025/awards_detail), and the released benchmarks (MediumDocEdit-Chat, BigCodeBench-Chat, MATH-Chat) give the community standard testbeds for collaborative behavior that prior single-turn benchmarks could not measure. The work also makes the case — empirically — that asking better questions beats answering faster, which cuts against a default optimization pressure in chat-tuned models.

**Future directions:** Opens research questions in (1) *user-simulator fidelity* — how realistic must a simulator be before MR transfers to real users, and can simulators be co-trained with the policy?, (2) *long-horizon credit assignment* — beyond a fixed forward window, how should agents reason about much longer collaborations spanning sessions and tools?, and (3) *personalization and adaptation* — extending MR to capture per-user preferences, expertise, and communication style as part of the reward.

## Collaborations

- **Academic:** Collaboration with Stanford University (Shirley Wu, Jure Leskovec, James Zou, Gavin Li, Weixin Cai) and Georgia Tech (Yao Dou) — co-authorship on the training framework, simulators, and benchmarks.
- **External / open source:** Code, models, benchmarks, and user simulators released publicly per the paper's Impact Statement; benchmarks built on top of community datasets (Medium, BigCodeBench, MATH, Abg-CoQA).

## Current status

**Headline:** CollabLLM yields **+18.5% task performance, +46.3% interactivity, and 13.3% greater efficiency** on average versus the strongest single-turn–trained baselines, and an **+17.6% user-satisfaction lift** in a 201-participant real-world study.

- **Simulated benchmarks (Online DPO over Llama-3.1-8B-Instruct):** MediumDocEdit-Chat BLEU 32.2 → 36.8 with interactivity 46 → 92; BigCodeBench-Chat pass rate 9.3 → 13.0 with interactivity 22 → 52; MATH-Chat accuracy 11.0 → 16.5 with average tokens reduced 3.40k → 2.37k.
- **User study (n=201, Amazon Mechanical Turk):** document quality avg **8.50/10**, **91.4%** of conversations rated "good" (8–9), **63.8%** rated CollabLLM highly engaging vs. 42.6% for the base model, and **~129 seconds** saved per task vs. the strongest baseline.
- **Recognition:** [ICML 2025 Outstanding Paper Award](https://icml.cc/virtual/2025/awards_detail).
- **Release:** Code, models, three Chat benchmarks (MediumDocEdit-Chat, BigCodeBench-Chat, MATH-Chat), and user simulators published at [aka.ms/CollabLLM](https://aka.ms/CollabLLM).

## Related landscape

- [Laban et al., *LLMs Get Lost In Multi-Turn Conversation* (arXiv 2505.06120, May 2025)](https://arxiv.org/abs/2505.06120) — Illustrates how today's LLMs fail in multi-turn settings (average 39% performance drop vs. single-turn), directly motivating the problem CollabLLM is built to solve.
- [*SimulatorArena: Are User Simulators Reliable Proxies for Multi-Turn Evaluation of AI Assistants?* (arXiv 2510.05444, Oct 2025)](https://arxiv.org/abs/2510.05444) — User simulation is an open problem, and SimulatorArena evaluates user simulators on how well they proxy real users — a methodological foundation for the simulator-driven training and evaluation CollabLLM relies on.
- [Naous, Laban, Xu, Neville — *UserLM-8b* (Hugging Face, Oct 2025)](https://huggingface.co/microsoft/UserLM-8b) — A more advanced class of user simulator, purpose-trained to play the user role with realism as the central focus, complementary to CollabLLM's reliance on simulator quality during training.

## Real-world impact

- ICML 2025 Outstanding Paper Award, recognizing the multi-turn-aware reward framework as a notable contribution to alignment research.
- Public release of code, model checkpoints, three new multi-turn benchmarks, and user simulators, lowering the barrier for the community to train and evaluate collaborative LLMs.
- Validated end-to-end with a 201-participant user study on real document-writing tasks — not just simulated rollouts — providing rare empirical evidence that multi-turn-aware training transfers to humans.

## Publications & links

- [CollabLLM: From Passive Responders to Active Collaborators — ICML 2025 (Outstanding Paper Award)](https://arxiv.org/abs/2502.00640)
- [MSR Blog — CollabLLM: Teaching LLMs to collaborate with users](https://www.microsoft.com/en-us/research/blog/collabllm-teaching-llms-to-collaborate-with-users/)
- [Project page: aka.ms/CollabLLM](https://aka.ms/CollabLLM)
- [MSR publication page](https://www.microsoft.com/en-us/research/publication/collabllm-from-passive-responders-to-active-collaborators/)
