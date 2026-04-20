# MedVQA-PEFT: 参数高效微调在医疗视觉问答中的前沿对比研究 (2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2026-orange.svg)](https://github.com/unslothai/unsloth)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue.svg)](https://huggingface.co)

## 📖 项目简介

随着大型多模态模型（LMMs）在医学影像理解中的突破，其高昂的部署与微调成本成为了在真实临床环境中落地的最大阻碍。本项目 (**MedVQA-PEFT-Bench**) 旨在提供一个极其严格的实证评估基准：**在硬件受限（≤ 24GB VRAM）的模拟临床环境中，哪种参数高效微调（PEFT）策略能实现医学视觉问答（Med-VQA）任务的性能最大化与幻觉最小化？**

我们首次在 Med-VQA 任务中，将传统方法（LoRA）与新一代微调算法（**DoRA, PiSSA**）进行了系统性对比，并创新性地构建了包含 **Med-POPE 幻觉探针** 和 **GPT-4 语义裁判** 的双轨制多维评估体系。

我们使用 **2026年最新原生多模态大模型**（Qwen3.5-9B 与 Gemma-4-E4B-IT），系统对比了三种前沿**参数高效微调（PEFT）方法**：**LoRA、DoRA、PiSSA**，在两个经典医疗视觉问答（Med-VQA）数据集上进行了全面实验。

**核心目标**：为资源受限的临床环境提供**高性能、低幻觉、可边缘部署**的Med-VQA解决方案。

### 主要创新与贡献
- 首次将 **Qwen3.5-9B**（Early-Fusion原生多模态）和 **Gemma-4-E4B-IT**（边缘优化多模态）应用于Med-VQA PEFT对比。
- 引入 **DoRA** 与 **PiSSA** 两种2025-2026前沿PEFT方法，远超原论文仅LoRA vs AdaLoRA的设置。
- 构建**多维度评价体系**：封闭式（Accuracy + POPE幻觉率） + 开放式（Keyword Recall + GPT-4o-as-a-Judge） + 完整部署效率指标。
- 支持**后期小样本泛化测试**，模拟真实临床跨数据集/跨医院场景。
- 全部实验均可在**单卡4090**上完成（Unsloth加速）。

## 🧬 模型详情

### 1. Qwen3.5-9B（主打性能）
- **HuggingFace**：`Qwen/Qwen3.5-9B`
- **参数规模**：9B（Dense）
- **架构特点**：Early-Fusion原生多模态，视觉token与文本token联合预训练
- **优势**：视觉空间推理、医学影像细粒度理解极强，中文医疗场景支持优秀
- **适用**：追求最高准确率与最低幻觉的临床场景

### 2. Gemma-4-E4B-IT（主打边缘部署）
- **HuggingFace**：`google/gemma-4-E4B-IT`
- **参数规模**：约4.5B有效参数（边缘优化版）
- **架构特点**：ViT视觉编码器 + 可变分辨率 + 原生音频支持
- **优势**：极低显存占用、推理速度快、适合手机/基层医院部署
- **适用**：强调临床实际部署效率的场景

## 🔧 PEFT方法详解

| 方法   | 全称                              | 核心创新点                          | 推荐rank | 显存优势          | 适用场景                  |
|--------|-----------------------------------|-------------------------------------|----------|-------------------|---------------------------|
| LoRA   | Low-Rank Adaptation              | 经典低秩矩阵适配                    | 16       | 中                | 稳定基线                  |     |
| DoRA   | Weight-Decomposed LoRA           | 权重分解为幅度+方向，只调方向       | 16       | 优秀              | 2026性能最优升级版        |
| PiSSA  | Principal Singular-value Adaptation | SVD主奇异向量微调，参数极少       | 16       | 优秀              | 参数极致压缩场景          |

**统一超参推荐**（本项目默认）：
- `rank=16`, `lora_alpha=32`, `lora_dropout=0.05`
- `target_modules` = vision projector + q_proj + k_proj + v_proj + gate_proj + up_proj + down_proj

  * **统一训练超参**：
      * `per_device_train_batch_size` = 1
      * `gradient_accumulation_steps` = 8 (等效 Batch Size = 8)
      * `learning_rate` = 1e-4 \~ 2e-4 (基于验证集 Loss 动态寻优)
      * `optimizer` = paged\_adamw\_32bit
      * 混合精度训练 = fp16 / bf16

> **⚠️ 内存优化机制**：所有基座模型默认采用 `BitsAndBytes` 框架下的 **4-bit NormalFloat (NF4)** 量化加载，配合双重量化（Double Quantization）与 Bfloat16 计算数据类型，将 9B 级别模型的静态显存占用压缩至 6GB 以下。
## 📚 数据集

- **VQA-RAD**：314张放射学影像，2244个QA对（radiology-centric）
- **SLAKE-VQA**：642张多模态医学影像（X-ray/CT/MRI），7033个QA对（multi-modality）

**问题类型分布**：
- 同时包含 **封闭式（Yes/No）** 和 **开放式（描述性）** 问题
- 训练阶段：**完整混合使用**（不预先分割）
- 测试阶段：**自动过滤**为 Closed-set / Open-set 两个子集

**下载链接**（已内置脚本自动下载）：
- VQA-RAD：https://huggingface.co/datasets/flaviagiammarino/vqa-rad
- SLAKE-VQA：https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english

## 🧪 实验设置（完整可复现）

### 实验组合
共 **16个主实验** + **小样本泛化测试**：
- 2个模型 × 4种PEFT × 2个数据集

### 训练配置
- **训练集划分**：80% train / 10% val / 10% test（随机分层）
- **训练参数**：
  - Epochs: 3–5
  - Learning Rate: 2e-4
  - Batch Size: 1 + Gradient Accumulation: 8
  - Optimizer: AdamW (weight decay=0.05)
  - Mixed Precision: FP16 / BF16
  - Quantization: QLoRA使用NF4 4-bit
- **硬件**：单卡 NVIDIA RTX 4090（24GB）可完成全部实验

### 后期小样本泛化测试
在主实验模型上使用1000–2000张自建临床影像（可自行标注）进行zero-shot / few-shot测试，评估跨数据集/跨医院泛化能力。

## 📏 评价指标（多维度）

### 封闭式问题（Closed-set / Yes/No）
- **Accuracy**（准确率）
- **POPE Hallucination Rate**（对象幻觉率）
  - 三个子集：Random / Popular / Adversarial（重点报告Adversarial）
  - 公式：`Hallucination Rate = (模型回答“Yes”但GT为“No”的比例) × 100%`

### 开放式问题（Open-set）
- **Keyword Recall**  
  公式：  
  \[
  \text{Keyword Recall} = \frac{\text{出现在生成答案中的 GT 关键词数量}}{\text{GT 关键词总数量}}
  \]
- **GPT-4o-as-a-Judge Semantic Score**（0-5分）
  - 使用医疗专用Prompt，重点考察事实一致性、临床专业性、幻觉

### 通用部署效率指标
- Training Time（秒）
- Peak VRAM（峰值显存占用，GB）
- Inference Speed（tokens/s 或 samples/s）
- Eval / Test Loss



基于两大主流医学影像数据集：**VQA-RAD** (放射学) & **SLAKE-VQA** (多模态/中英文)。

**独创的数据流转策略（Stratified Pipeline）：**

  * **训练集/验证集 (Train/Val)**：将封闭式（Yes/No）与开放式（Open-ended）问答彻底混合打乱，保障模型通用指令遵循能力，防止“答题模式”过拟合。
  * **测试集 (Test)**：严格保持纯净。推理结束后，脚本自动根据题型进行**分流（Split）**，防止封闭式问题的高分掩盖开放式问题的幻觉。

-----

## 📏 多维度评价指标 (Multi-dimensional Evaluation Metrics)

我们抛弃了传统且僵化的 BLEU/ROUGE 分数，建立了包含 3 个维度、5 大指标的现代医疗 AI 评价体系：

### 1\. 基础事实与推理能力 (Factual Accuracy)

  * **Yes/No Accuracy**: 针对封闭式问题的精确匹配准确率（Exact Match）。
  * **Open-ended Recall**: 针对开放式回答，计算基准答案（Ground-Truth）关键词在生成文本中的召回率。

### 2\. 生成质量与语义一致性 (Semantic Quality)

  * **GPT-4-as-a-Judge Score**:
      * 摒弃字面匹配，调用文本版 GPT-4 对生成的开放式长答案进行医疗事实核查。
      * **评分量表**：0 分（完全矛盾/错误），1 分（部分正确/缺乏细节），2 分（语义与临床事实完全一致）。

### 3\. 医学幻觉自测体系 (Hallucination Detection)

  * **Med-POPE (Polling-based Object Probing Evaluation)**:
      * 基于 SLAKE 数据集的 Bounding Box 标注，自动生成探针问题（如：*“图中有肝脏肿瘤吗？”*）。
      * 包含三个诱导难度等级：**Random**（随机物体）、**Popular**（高频存在但本图缺失的物体）、**Adversarial**（临床常伴生但本图缺失的物体）。
      * **核心指标**：Precision（精确率）、Yes-Ratio（模型盲目回答 Yes 的“讨好比例”）。

### 4\. 系统能效 (System Efficiency)

  * **Peak VRAM**: 训练期间的显存占用峰值（MB），通过 WandB 实时捕获。
  * **Training Time**: 达到验证集 Loss 收敛点的时间成本。

-----


**所有指标均自动记录到 wandb + results/ 目录**。

## 🛠️ 项目结构

```
MedVQA-PEFT/
├── data/                  # 数据集下载与预处理
├── configs/               # 各PEFT方法的完整超参配置文件
├── models/                # 模型加载与PEFT wrapper
├── scripts/
│   ├── train.py           # 一键训练主脚本
│   ├── evaluate.py        # 完整指标评估（支持所有指标）
│   ├── pope_eval.py
│   └── gpt_judge.py
├── utils/
│   ├── metrics.py         # Keyword Recall + GPT Judge实现
│   ├── data_utils.py      # Closed/Open自动过滤
│   └── visualization.py
├── results/               # 实验结果表格（Markdown + CSV + Excel）
├── notebooks/             # Colab一键运行笔记本（推荐）
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/lkxgx/MedVQA.git
cd MedVQA

# 2. 安装依赖（推荐Unsloth）
pip install -r requirements.txt

# 3. 训练示例（Qwen3.5-9B + DoRA）
python scripts/train.py \
  --model_name Qwen/Qwen3.5-9B \
  --peft_method DoRA \
  --dataset both \
  --output_dir outputs/qwen3.5-9b-dora

# 4. 完整评估
python scripts/evaluate.py \
  --checkpoint outputs/qwen3.5-9b-dora \
  --dataset VQA-RAD \
  --run_pope --run_gpt_judge
```

## 📈 实验结果（持续更新）

实验结果将自动生成在 `results/` 目录，目前已规划16个主实验 + 泛化测试。  

## 📜 引用

本项目对应论文（投稿中）：
> 《Qwen3.5与Gemma-4参数高效微调前沿方法对比研究：LoRA、QLoRA、DoRA与PiSSA在医疗视觉问答中的幻觉缓解与临床部署优化》

**BibTeX**（发表后更新）。

## 🙏 致谢

- 基准论文：Rezaei et al. (Computers in Biology and Medicine, 2026)
- 工具支持：Unsloth, Hugging Face, VLMEvalKit, OpenAI
- 数据集：VQA-RAD & SLAKE-VQA 官方维护者

---

**欢迎 Star & Fork！**  
有任何疑问、想合作复现、wandb配置、论文LaTeX模板，随时在 Issues 提出。

**作者**：Kaixuan（2026年4月）  
**联系**：3193888648@qq.com