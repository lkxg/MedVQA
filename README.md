# MedVQA-PEFT: A Comparative Study of Parameter-Efficient Fine-Tuning for Medical Visual Question Answering

[English](README.md) | [简体中文](README.zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2026-orange.svg)](https://github.com/unslothai/unsloth)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue.svg)](https://huggingface.co)

## Project Overview

Large multimodal language models (MLLMs) have significantly advanced medical image understanding, but deployment and fine-tuning costs remain a major barrier in real clinical settings. This project, **MedVQA-PEFT**, provides a reproducible benchmark to answer a practical question:

**Under constrained hardware (<= 24GB VRAM), which parameter-efficient fine-tuning (PEFT) strategy achieves stronger Med-VQA performance with lower hallucination rates?**

We evaluate two natively multimodal base models (Qwen3.5-9B and Gemma-4-E4B-IT) across three PEFT methods: **LoRA, DoRA, and PiSSA**. The evaluation protocol combines a **Med-POPE hallucination probe** and **GPT-5 semantic judging** for multi-dimensional analysis.

Core objective: deliver high-accuracy, low-hallucination, deployment-ready Med-VQA solutions for resource-limited clinical environments.

### Key Contributions

- Systematic PEFT comparison for Med-VQA: **LoRA / DoRA / PiSSA**.
- Multi-dimensional evaluation: closed-set (Accuracy + POPE hallucination rate), open-set (Keyword Recall + GPT-5-as-a-Judge), and deployment efficiency metrics.
- Cross-dataset generalization tests under zero-shot and few-shot settings.
- Reproducible single-GPU target setup (RTX 4090, 24GB VRAM).

## Architecture Choice: Adapter-Style VL vs Natively Multimodal

Modern multimodal models generally follow two architecture paradigms. This project intentionally uses **natively multimodal models** (Qwen3.5 / Gemma-4) as backbones.

### Adapter-Style VL Models (Late Fusion / Bolt-on Vision)

Examples: LLaVA family, Qwen-VL / Qwen2-VL / Qwen3-VL, LLaMA-3.2-Vision, MiniGPT-4, InstructBLIP

- Architecture: `pretrained vision encoder (CLIP / SigLIP ViT) + projection layer (MLP or cross-attention) + pretrained LLM`
- Training: two-stage alignment, then visual instruction tuning
- Fusion: image tokens are projected and prepended to text tokens at input time
- Pros: modular design, easy vision encoder replacement, strong LLM reuse
- Cons: modality alignment gap, longer context pressure from visual tokens, weaker fine-grained spatial reasoning

### Natively Multimodal Models (Early Fusion)

Examples: Qwen3.5, Gemma-4, Gemini, GPT-4o

- Architecture: a unified Transformer stack jointly processes visual and text tokens from pretraining onward
- Training: mixed image-text-video pretraining without an explicit separate alignment phase
- Fusion: deep cross-modal interaction at every layer
- Pros: more unified representations, better parameter efficiency (especially smaller models), stronger fine-grained VQA behavior
- Cons: less modularity for swapping vision subcomponents, higher pretraining cost from scratch

### Quick Comparison

| Dimension | Adapter-Style VL | Natively Multimodal |
|---|---|---|
| Coupling | Vision + LLM are separable | End-to-end unified stack |
| Alignment | Mandatory and often a bottleneck | Learned during pretraining |
| Vision-language interaction | Shallow (input-side) | Deep (full-stack) |
| Parameter efficiency | Lower in practice | Higher, especially for compact models |
| Hallucination tendency | Modality gap can increase hallucinations | Joint modeling tends to improve control |
| Edge deployment | Vision encoder can dominate cost | More compact end-to-end profile |
| Typical examples | LLaVA, Qwen3-VL, LLaMA-3.2-V | **Qwen3.5, Gemma-4** |

Motivation: medical VQA requires robust fine-grained visual reasoning and hallucination control. Natively multimodal models are better aligned with these requirements while remaining feasible on a single 24GB GPU.

## Model Details

### 1) Qwen3.5-9B (Performance-Oriented)

- Hugging Face: `Qwen/Qwen3.5-9B`
- Release: 2026-03-02 (9B Dense variant; Qwen3.5 family initial release 2026-02-16)
- Parameters: 9B (Dense)
- Highlights: hybrid **Gated DeltaNet + Gated Attention**; native early-fusion multimodality
- Context length: native 262K, RoPE-extended beyond 1M
- Strengths: strong joint visual-text reasoning, OCR/chart/spatial tasks
- Best fit: scenarios prioritizing top accuracy and reduced hallucinations

### 2) Gemma-4-E4B-IT (Deployment-Oriented)

- Hugging Face: `google/gemma-4-E4B-it` (instruction-tuned); base repo `google/gemma-4-E4B`
- Release: 2026-04-02 (Apache 2.0)
- Parameters: **4B effective active parameters** (E = Effective)
- Highlights: **Per-Layer Embeddings (PLE)** + alternating local/global attention; native support for text/image/video, and audio input for E4B
- Context length: 128K
- Strengths: low memory footprint, faster inference, configurable visual token budget (70-1120 tokens/image)
- Best fit: deployment efficiency and broad multimodal coverage

## PEFT Methods

| Method | Full Name | Core Idea | Recommended Rank | Memory Advantage | Typical Use |
|---|---|---|---|---|---|
| LoRA | Low-Rank Adaptation | Classic low-rank adaptation baseline | 16 | Medium | Stable reference baseline |
| DoRA | Weight-Decomposed LoRA | Decompose weight into magnitude and direction; optimize direction more explicitly | 16 | High | Performance-first |
| PiSSA | Principal Singular-value Adaptation | Principal singular-vector guided initialization/adaptation | 16 | High | Parameter-compression priority |

Default hyperparameters:

- `rank=16`, `lora_alpha=32`, `lora_dropout=0.05`
- `target_modules`: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Training defaults:
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=8` (effective batch size = 8)
  - `learning_rate=2e-4` (usually tuned in `1e-4 ~ 2e-4`)
  - `optimizer=paged_adamw_32bit`
  - mixed precision: `fp16` or `bf16`

Memory optimization: NF4 4-bit quantization via BitsAndBytes is enabled by default (optionally with double quantization and bf16 compute) to reduce static VRAM usage.

## Datasets

### In-Domain Main Experiments

- **VQA-RAD**: 314 radiology images, 2244 QA pairs
- **SLAKE-VQA**: 642 medical images, 7033 QA pairs

### Out-of-Domain Generalization

- **PMC-VQA-test-clean**: a cleaned subset from PMC-VQA with broader modalities and specialties for zero-shot/few-shot transfer evaluation

Important: models are **not trained on PMC-VQA**. It is only used for zero-shot/few-shot inference.

### Stratified QA Pipeline

- Train/validation: mixed closed-set (yes/no) + open-set questions
- Test: auto-routed by question type into closed-set and open-set evaluation pipelines

Dataset links:

- VQA-RAD: https://huggingface.co/datasets/flaviagiammarino/vqa-rad
- SLAKE-VQA: https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english
- PMC-VQA-test-clean: https://huggingface.co/datasets/RadGenome/PMC-VQA (`test-clean` split)

## Experimental Setup (Reproducible)

### Experiment Matrix

12 main experiments + cross-dataset generalization:

- `2 backbones x 3 PEFT methods x 2 datasets`

### Training Configuration

- Data split policy:
  - SLAKE uses official `train / validation / test`
  - VQA-RAD provides `train / test`; we split 10% from training as validation with fixed `seed=42`
- Epochs: 3-5
- Learning rate: 2e-4
- Batch setup: batch size 1 + gradient accumulation 8
- Optimizer: paged_adamw_32bit
- Mixed precision: FP16 / BF16
- Quantization: NF4 4-bit
- Hardware: single NVIDIA RTX 4090 (24GB)

### Cross-Dataset Evaluation on PMC-VQA-test-clean

After the 12 main training runs, checkpoints are evaluated on PMC-VQA-test-clean in zero-shot/few-shot mode to measure distribution-shift robustness and hallucination drift.

Metrics stay aligned with main experiments: Accuracy / POPE / Keyword Recall / GPT-5 Judge.

## Evaluation Metrics

### Closed-Set Questions (Yes/No)

- Accuracy
- Med-POPE hallucination rate:
  - Constructed from yes/no anatomy/lesion existence questions
  - Three negative-sample settings:
    - Random: random unseen object replacement
    - Popular: high-frequency anatomy replacement
    - Adversarial: highest co-occurrence and most confusable anatomy replacement

Formula:

`Hallucination Rate = P(Answer=Yes | GT=No) = FP / (FP + TN)`

### Open-Set Questions

- Keyword Recall

\[
\text{Keyword Recall} = \frac{\text{# matched GT keywords in generated answer}}{\text{# total GT keywords}}
\]

- GPT-5-as-a-Judge semantic score (0-5): factual consistency, clinical correctness, and hallucination behavior

### System Efficiency Metrics

- Training time (seconds)
- Peak VRAM (GB)
- Inference speed (tokens/s or samples/s)
- Eval/test loss

All metrics can be tracked in Weights & Biases and exported to `results/`.

## Repository Structure

```text
MedVQA/
├── data/
│   ├── slake/
│   └── vqa-rad/
├── models/
│   ├── Qwen3.5-9B/
│   └── gemma-4-E4B-it/
├── outputs/
├── scripts/
│   ├── download_datasets.py
│   └── download_models.py
├── configs/
├── utils/
├── requirements.txt
├── README.md
├── README.zh.md
└── LICENSE
```

## Quick Start

```bash
# 1) Clone
git clone https://github.com/lkxgx/MedVQA.git
cd MedVQA

# 2) Install dependencies
pip install -r requirements.txt

# 3) Download datasets
python scripts/download_datasets.py

# 4) Download models
#    Note: for Gemma, first accept Google terms on:
#    https://huggingface.co/google/gemma-4-E4B-it
#    then provide an access token.
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

python scripts/download_models.py --model all --hf_token $HF_TOKEN

# Or download separately
python scripts/download_models.py --model qwen3.5-9b
python scripts/download_models.py --model gemma-4-e4b --hf_token $HF_TOKEN
```

## Results

It is recommended to store experiment outputs under `results/` (CSV, Markdown, and plots/reports).

## Citation

Paper in submission:

> "Comparative PEFT Study of Qwen3.5 and Gemma-4 for Medical VQA: Hallucination Mitigation and Deployment Optimization with LoRA, DoRA, and PiSSA"

BibTeX will be added after publication.

## Acknowledgments

- Tooling: Unsloth, Hugging Face, OpenAI
- Datasets: VQA-RAD and SLAKE-VQA maintainers

Contributions, issues, and reproducibility discussions are welcome.

Author: Kaixuan (April 2026)
Contact: 3193888648@qq.com