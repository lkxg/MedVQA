#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_gemma4.py
Gemma-4-E4B-IT 专用训练脚本（优化版）
"""

import unsloth
import argparse
import wandb
from pathlib import Path
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator

DATASET_ALIASES = {
    "VQA-RAD": "vqa-rad",
    "SLAKE-VQA": "slake",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma-4-E4B-IT Med-VQA 训练脚本（优化版）")
    
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["VQA-RAD", "SLAKE-VQA"], required=True)
    parser.add_argument("--data_root", type=str, default="data/processed")

    parser.add_argument("--peft", type=str, choices=["lora", "dora", "pissa"], default="lora")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="outputs/gemma4-e4b")
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="MedVQA-PEFT-Gemma4")
    parser.add_argument("--wandb_name", type=str, default=None)

    return parser.parse_args()


def apply_peft(model, args):
    kwargs = {
        "r": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "target_modules": "all-linear",
    }

    if args.peft == "lora":
        model = FastVisionModel.get_peft_model(model, **kwargs)
    elif args.peft == "dora":
        model = FastVisionModel.get_peft_model(model, use_dora=True, **kwargs)
    elif args.peft == "pissa":
        model = FastVisionModel.get_peft_model(model, init_lora_weights="pissa", **kwargs)
    return model


def main():
    args = parse_args()
    run_name = f"gemma4-e4b-{args.dataset.lower()}-{args.peft}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        try:
            wandb.init(project=args.wandb_project, name=args.wandb_name or run_name, config=vars(args))
            report_to = "wandb"
        except Exception:
            print("⚠️ wandb 初始化失败，已关闭")
            report_to = "none"
    else:
        report_to = "none"

    print(f"🚀 开始实验: {run_name}")

    # 加载模型
    if args.model_path and Path(args.model_path).exists():
        model, tokenizer = FastVisionModel.from_pretrained(args.model_path, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    else:
        model, tokenizer = FastVisionModel.from_pretrained("google/gemma-4-E4B-IT", load_in_4bit=True, use_gradient_checkpointing="unsloth")

    model = apply_peft(model, args)
    FastVisionModel.for_training(model)

    # 加载数据集（同上）
    train_path = Path(args.data_root) / DATASET_ALIASES[args.dataset] / "train"
    val_path = Path(args.data_root) / DATASET_ALIASES[args.dataset] / "validation"

    train_dataset = load_from_disk(str(train_path))
    val_dataset = load_from_disk(str(val_path))

    data_collator = UnslothVisionDataCollator(model, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=str(output_dir),
            report_to=report_to,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    print("开始训练...")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"\n✅ 训练完成！Adapter 保存至: {adapter_dir}")
    if report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()