#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py
MedVQA-PEFT 项目完整评估脚本
支持加载 Unsloth PEFT checkpoint，评估 VQA-RAD / SLAKE-VQA / PMC-VQA-test-clean
"""

import unsloth
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import wandb
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastVisionModel

# ====================== 评估专用依赖 ======================
from vlmeval import POPE  # POPE 幻觉率
import openai
from openai import OpenAI

# ====================== 配置 ======================
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # 请提前设置环境变量

def is_closed_set(answer: str) -> bool:
    """判断是否为封闭式问题（Yes/No）"""
    clean = str(answer).strip().lower()
    return clean in ["yes", "no", "是", "否", "有", "无", "true", "false"] or len(clean.split()) <= 4

def keyword_recall(gt_text: str, pred_text: str) -> float:
    """简单关键词召回率（开放式问题）"""
    gt_words = set(str(gt_text).lower().split())
    pred_words = set(str(pred_text).lower().split())
    if not gt_words:
        return 1.0
    return len(gt_words & pred_words) / len(gt_words)

def gpt4_judge(gt: str, pred: str) -> dict:
    """GPT-4o Judge 语义评分"""
    prompt = f"""你是一位资深放射科医生，正在评估医疗视觉问答的答案质量。
问题答案：
标准答案: {gt}
模型生成: {pred}

请严格按0-5分打分（5分=完全正确无幻觉，0分=严重幻觉）：
输出严格JSON格式：
{{"score": 5, "reason": "详细理由", "hallucination": false}}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"GPT Judge 失败: {e}")
        return {"score": 0, "reason": "API error", "hallucination": True}

def main():
    parser = argparse.ArgumentParser(description="MedVQA-PEFT 评估脚本")
    parser.add_argument("--checkpoint", type=str, required=True, help="微调后checkpoint路径，例如 outputs/qwen35-9b-dora-slake/final_model")
    parser.add_argument("--dataset", type=str, choices=["VQA-RAD", "SLAKE-VQA", "PMC-VQA-test-clean"], default="VQA-RAD")
    parser.add_argument("--wandb_project", type=str, default="MedVQA-PEFT-Eval")
    parser.add_argument("--max_samples", type=int, default=None, help="调试用，限制评估样本数")
    args = parser.parse_args()

    exp_name = Path(args.checkpoint).parent.name
    wandb.init(project=args.wandb_project, name=f"eval-{exp_name}-{args.dataset}", config=vars(args))

    # 1. 加载模型（Unsloth PEFT checkpoint）
    print(f"🚀 加载模型: {args.checkpoint}")
    model, processor = FastVisionModel.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=True,
    )
    model = FastVisionModel.for_inference(model)   # 切换为推理模式

    # 2. 加载测试集
    print(f"📥 加载数据集: {args.dataset}")
    if args.dataset == "VQA-RAD":
        dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")
    elif args.dataset == "SLAKE-VQA":
        dataset = load_dataset("mdwiratathya/SLAKE-vqa-english", split="test")
    else:  # PMC-VQA-test-clean
        dataset = load_dataset("TIGER-Lab/PMC-VQA", split="test")   # 实际可能需要调整为具体clean子集

    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    # 3. 开始评估
    results = []
    closed_preds, closed_gts = [], []
    open_preds, open_gts = [], []

    start_time = time.time()

    for sample in tqdm(dataset, desc="评估中"):
        image = sample["image"]
        question = sample["question"]
        gt = sample["answer"]

        # 生成答案
        inputs = processor(image, question, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128)
        pred = processor.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "question": question,
            "gt": gt,
            "pred": pred,
            "is_closed": is_closed_set(gt)
        })

        if is_closed_set(gt):
            closed_gts.append(gt)
            closed_preds.append(pred)
        else:
            open_gts.append(gt)
            open_preds.append(pred)

    total_time = time.time() - start_time

    # 4. 计算指标
    # Closed-set
    closed_acc = sum(1 for p, g in zip(closed_preds, closed_gts) if p.lower().strip() == g.lower().strip()) / len(closed_gts) if closed_gts else 0

    # POPE 幻觉率（只在 closed-set 上跑）
    pope_results = POPE.eval(model, dataset=args.dataset, split="adversarial") if closed_gts else {}

    # Open-set
    open_recall = sum(keyword_recall(gt, pred) for gt, pred in zip(open_gts, open_preds)) / len(open_gts) if open_gts else 0

    gpt_scores = []
    for gt, pred in tqdm(zip(open_gts, open_preds), desc="GPT Judge"):
        res = gpt4_judge(gt, pred)
        gpt_scores.append(res["score"])

    avg_gpt_score = sum(gpt_scores) / len(gpt_scores) if gpt_scores else 0

    # 5. 保存结果
    df = pd.DataFrame(results)
    result_dir = Path(args.checkpoint).parent / "eval_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(result_dir / f"{args.dataset}_results.csv", index=False)
    df.to_markdown(result_dir / f"{args.dataset}_results.md")

    # 6. wandb 记录
    wandb.log({
        f"{args.dataset}/closed_accuracy": closed_acc,
        f"{args.dataset}/open_keyword_recall": open_recall,
        f"{args.dataset}/gpt4_judge_score": avg_gpt_score,
        f"{args.dataset}/pope_hallucination_rate": pope_results.get("hallucination_rate", 0),
        f"{args.dataset}/inference_time_seconds": total_time,
    })

    print("\n" + "="*60)
    print(f"✅ 评估完成！数据集: {args.dataset}")
    print(f"Closed Accuracy: {closed_acc:.4f}")
    print(f"Open Keyword Recall: {open_recall:.4f}")
    print(f"GPT-4 Judge Score: {avg_gpt_score:.2f}/5")
    print(f"POPE Hallucination Rate: {pope_results.get('hallucination_rate', 0):.4f}")
    print(f"结果已保存至: {result_dir}")
    print("="*60)

    wandb.finish()

if __name__ == "__main__":
    main()