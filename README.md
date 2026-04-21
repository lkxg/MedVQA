# MedVQA-PEFT: 参数高效微调在医疗视觉问答中的对比研究

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2026-orange.svg)](https://github.com/unslothai/unsloth)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue.svg)](https://huggingface.co)

## 📖 项目简介

随着大型多模态模型（MLLMs）在医学影像理解中的突破，其高昂的部署与微调成本成为了在真实临床环境中落地的主要阻碍。本项目（**MedVQA-PEFT**）旨在提供一个可复现的实证评估基准：**在硬件受限（≤24GB VRAM）的模拟临床环境中，哪种参数高效微调（PEFT）策略能在 Med-VQA 任务上实现更高性能与更低幻觉率。**

我们使用两种原生多模态模型（Qwen3.5-9B 与 Gemma-4-E4B-IT），系统对比三种前沿 PEFT 方法：**LoRA、DoRA、PiSSA**，并构建了包含 **Med-POPE 幻觉探针** 与 **GPT-5 语义裁判** 的多维评估体系。

**核心目标**：为资源受限的临床环境提供高性能、低幻觉、可边缘部署的 Med-VQA 方案。

### 主要贡献

- 在 Med-VQA 场景中系统比较 **LoRA / DoRA / PiSSA** 三种 PEFT 路线。
- 构建多维评估体系：封闭式（Accuracy + POPE 幻觉率）+ 开放式（Keyword Recall + GPT-5-as-a-Judge）+ 部署效率指标。
- 支持小样本后测（few-shot / zero-shot）以模拟跨数据集、跨医院泛化。
- 目标硬件为单卡 4090（24GB），强调可复现与部署可行性。

## 🆚 架构对比：适配器式 VL vs 原生多模态

业内多模态大模型大致可分为两类架构范式。本项目刻意选择**原生多模态模型**（Qwen3.5 / Gemma-4）作为评测底座，以下对比说明选型动机。

### 适配器式 VL 模型（Late-Fusion / Bolt-on Vision）

**代表**：LLaVA 系列、Qwen-VL / Qwen2-VL / Qwen3-VL、LLaMA-3.2-Vision、MiniGPT-4、InstructBLIP

- **架构**：`预训练视觉编码器 (CLIP / SigLIP ViT) + 投影层 (MLP 或 Cross-Attention) + 预训练 LLM`
- **训练**：两阶段 —— (1) 图文对齐预训练（冻结 LLM，只训投影层）；(2) 视觉指令微调
- **模态融合**：视觉 token 经投影后拼接到文本序列前端，融合发生在 LLM 推理时
- **优点**：模块化强、可替换视觉编码器、可复用强力预训练 LLM
- **缺点**：存在模态对齐 gap、视觉 token 占用长上下文、细粒度空间推理相对较弱

### 原生多模态模型（Natively Multimodal / Early-Fusion）

**代表**：Qwen3.5、Gemma-4、Gemini、GPT-4o

- **架构**：统一 Transformer 栈从预训练起就同时处理视觉与文本 token（共享 tokenizer 或联合 embedding 空间）
- **训练**：预训练全程混合图像–文本–视频数据，无显式对齐阶段
- **模态融合**：图像 patch 与文本 token 在每一层都深度交互（Early Fusion）
- **优点**：跨模态表征更统一、小参数量下仍具竞争力、细粒度视觉问答能力更强
- **缺点**：不便替换视觉子模块、从零预训练成本高

### 对比速览

| 维度 | 适配器式 VL | 原生多模态 |
|---|---|---|
| 架构耦合 | 视觉+LLM 可拆分 | 端到端统一栈 |
| 对齐阶段 | 必须，易成瓶颈 | 预训练即对齐 |
| 视觉–语言交互 | 浅层（输入端） | 深层（全层级） |
| 参数效率 | 相对较低 | 更高（尤其小模型） |
| 幻觉表现 | 模态 gap → 易幻觉 | 联合建模 → 相对可控 |
| 边缘部署 | 视觉编码器常为瓶颈 | 整体更紧凑 |
| 代表 | LLaVA, Qwen3-VL, LLaMA-3.2-V | **Qwen3.5, Gemma-4** |

**选型动机**：医疗影像问答对细粒度视觉推理与幻觉控制要求极高，原生多模态的深层融合特性契合该需求；同时 Gemma-4-E4B 体积小、适合 24GB 单卡训练与边缘落地，构成“性能 + 部署”双基线。

## 🧬 模型详情

### 1. Qwen3.5-9B（主打性能）

- **HuggingFace**：`Qwen/Qwen3.5-9B`
- **发布日期**：2026-03-02（9B Dense 变体；Qwen3.5 主系列 2026-02-16 首发）
- **参数规模**：9B（Dense）
- **架构亮点**：Hybrid **Gated DeltaNet + Gated Attention**；原生多模态 Early Fusion，预训练使用 trillions 级交错图–文–视频 token；与 Qwen-VL / Qwen2-VL / Qwen3-VL 的 late-fusion 路线形成代际切换
- **上下文长度**：原生 262K，RoPE 可扩展至 1M+
- **特点**：细粒度视觉–文本联合推理能力强，OCR / 图表 / 空间推理表现突出
- **适用**：追求更高准确率与更低幻觉率的场景

### 2. Gemma-4-E4B-IT（主打边缘部署）

- **HuggingFace**：`google/gemma-4-E4B-it`（Instruction-Tuned；Base 仓库为 `google/gemma-4-E4B`）
- **发布日期**：2026-04-02（Apache 2.0）
- **参数规模**：**4B effective active parameters（E = Effective）** —— 推理时实际激活约 4B，借助 Per-Layer Embeddings 压缩静态显存
- **架构亮点**：**Per-Layer Embeddings (PLE)** + 交替局部滑窗 / 全局注意力；原生支持 text / image / video，**E4B 额外原生支持 audio 输入**（无需独立 STT）
- **上下文长度**：128K
- **特点**：显存占用低、推理速度快，支持可配置视觉 token 预算（70–1120 tokens/image）在细节与速度间折中
- **适用**：强调实际部署效率与跨模态覆盖的场景

## 🔧 PEFT 方法

| 方法 | 全称 | 核心思路 | 推荐 rank | 显存优势 | 适用场景 |
|---|---|---|---|---|---|
| LoRA | Low-Rank Adaptation | 经典低秩适配基线 | 16 | 中 | 稳定对照组 |
| DoRA | Weight-Decomposed LoRA | 权重分解为幅度与方向，重点优化方向 | 16 | 优秀 | 性能优先 |
| PiSSA | Principal Singular-value Adaptation | 基于主奇异向量初始化/适配，参数效率高 | 16 | 优秀 | 参数压缩优先 |

**统一超参建议（默认）**：

- `rank=16`, `lora_alpha=32`, `lora_dropout=0.05`
- `target_modules`：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`（attention + MLP 全部投影层；原生多模态视觉与文本共享统一栈、无独立投影层，具体模块名因模型而异，以 `configs/qwen3_5.yaml` 和 `configs/gemma_4.yaml` 为准）
- 训练超参：
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=8`（等效 batch size=8）
  - `learning_rate=2e-4`（可在 `1e-4 ~ 2e-4` 内调优）
  - `optimizer=paged_adamw_32bit`
  - 混合精度：`fp16/bf16`

> **显存优化**：默认采用 BitsAndBytes 的 NF4 4-bit 量化（可配合 double quantization + bf16 计算），用于降低静态显存占用。

## 📚 数据集

### 主实验（In-domain 训练 + 测试）

- **VQA-RAD**：314 张放射学影像，2244 个 QA 对
- **SLAKE-VQA**：642 张多模态医学影像，7033 个 QA 对

### 跨数据集泛化测试（Out-of-domain, Zero-/Few-shot）

- **PMC-VQA-test-clean**：来自 PubMed Central 文献插图的大规模医学 VQA 测试集（清洗版），涵盖更广的影像模态与科室，用于评估主实验模型在未见分布上的泛化与幻觉表现。模型**完全不在 PMC-VQA 上训练**，仅做 zero-shot / few-shot 推理。

**题型策略（Stratified Pipeline）**：

- 训练/验证阶段：封闭式（Yes/No）与开放式（Open-ended）混合训练
- 测试阶段：按题型自动分流为 Closed-set / Open-set 分别评估

**下载链接**：

- VQA-RAD：https://huggingface.co/datasets/flaviagiammarino/vqa-rad
- SLAKE-VQA：https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english
- PMC-VQA-test-clean：https://huggingface.co/datasets/RadGenome/PMC-VQA （使用其 `test-clean` 子集）

## 🧪 实验设置（可复现）

### 实验组合

共 **12 个主实验** + 跨数据集泛化测试：

- `2 个模型 × 3 种 PEFT × 2 个数据集`

### 训练配置

- 数据划分：**遵循数据集官方划分** —— SLAKE 使用官方 `train / validation / test`；VQA-RAD 官方仅提供 `train / test`，从 `train` 中按固定随机种子（`seed=42`）抽取 10% 作为 `validation`，以保证可复现与跨论文可比
- Epochs：3-5
- Learning Rate：2e-4
- Batch Size：1 + Gradient Accumulation：8
- Optimizer：paged_adamw_32bit
- Mixed Precision：FP16 / BF16
- Quantization：NF4 4-bit
- 硬件：单卡 NVIDIA RTX 4090（24GB）

### 跨数据集泛化测试（PMC-VQA-test-clean）

在 12 个主实验训练完成的检查点基础上，使用 **PMC-VQA-test-clean** 进行 zero-shot / few-shot 推理，衡量模型跨数据源（VQA-RAD / SLAKE → PubMed 文献图）的泛化能力与幻觉漂移。评估指标与主实验对齐（Accuracy / POPE / Keyword Recall / GPT-5 Judge）。

## 📏 评价指标（多维度）

### 封闭式问题（Closed-set / Yes-No）

- **Accuracy**（准确率）
- **Med-POPE Hallucination Rate**（对象幻觉率，POPE 在医学场景下的适配版）
  - **构造方式**：从 VQA-RAD / SLAKE 的封闭式 QA 中抽取"是否存在某解剖结构 / 病灶"的 Yes/No 问句，并按 POPE 原始协议重构三档负样本：
    - **Random**：随机替换为训练集中未出现的对象 / 解剖结构
    - **Popular**：替换为在数据集中高频出现的解剖结构
    - **Adversarial**：替换为与原对象解剖学共现频率最高的"最易混淆"结构（重点报告）
  - **公式**：
    `Hallucination Rate = P(Answer=Yes | GT=No) = FP / (FP + TN)`
    即：在所有正确答案为 "No" 的样本中，模型回答 "Yes" 的比例

### 开放式问题（Open-set）

- **Keyword Recall**

\[
\text{Keyword Recall} = \frac{\text{生成答案中命中的 GT 关键词数}}{\text{GT 关键词总数}}
\]

- **GPT-5-as-a-Judge Semantic Score**（0-5 分）
  - 使用医疗场景评分 Prompt，关注事实一致性、临床专业性与幻觉情况

### 系统效率指标

- Training Time（秒）
- Peak VRAM（GB）
- Inference Speed（tokens/s 或 samples/s）
- Eval / Test Loss

**所有指标可记录到 wandb 与 `results/` 目录。**

## 🛠️ 当前仓库结构

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
└── LICENSE
```

## 🚀 快速开始

```bash
# 1) 克隆仓库
git clone https://github.com/lkxgx/MedVQA.git
cd MedVQA

# 2) 安装依赖
pip install -r requirements.txt

# 3) 下载数据集
python scripts/download_datasets.py

# 4) 下载模型
#    注意：Gemma 系列需先在 HF 页面（https://huggingface.co/google/gemma-4-E4B-it）
#    接受 Google 使用条款，并获取 Access Token。
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

python scripts/download_models.py --model all --hf_token $HF_TOKEN
# 或单独下载
python scripts/download_models.py --model qwen3.5-9b
python scripts/download_models.py --model gemma-4-e4b --hf_token $HF_TOKEN
```

## 📈 实验结果

实验结果建议统一保存到 `results/`（CSV/Markdown/可视化报告）。

## 📜 引用

本项目对应论文（投稿中）：

> 《Qwen3.5 与 Gemma-4 参数高效微调方法对比：LoRA、DoRA 与 PiSSA 在医疗视觉问答中的幻觉缓解与部署优化》

**BibTeX**：发表后补充。

## 🙏 致谢

- 工具支持：Unsloth, Hugging Face, OpenAI
- 数据集：VQA-RAD & SLAKE-VQA 官方维护者

---

欢迎 Star / Fork，欢迎 Issue 交流复现与改进建议。

**作者**：Kaixuan（2026年4月）
**联系**：3193888648@qq.com
