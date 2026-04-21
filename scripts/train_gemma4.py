#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_gemma4.py
Gemma-4-E4B-IT 专用训练脚本（支持 LoRA / DoRA / PiSSA）
"""

import argparse
import wandb
from pathlib import Path
from datasets import load_dataset
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth import SFTTrainer
from trl import SFTConfig

def main():
    parser = argparse.ArgumentParser(description="Gemma-4-E4B-IT Med-VQA 训练脚本")
    parser.add_argument("--peft", type=str, choices=["lora", "dora", "pissa"], default="lora", help="PEFT 方法")
    parser.add_argument("--dataset", type=str, choices=["VQA-RAD", "SLAKE-VQA"], default="VQA-RAD")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs/gemma4-e4b")
    args = parser.parse_args()

    exp_name = f"gemma4-e4b-{args.peft}-{args.dataset.lower()}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project="MedVQA-PEFT-Gemma4", name=exp_name, config=vars(args))

    print(f"🚀 开始实验: {exp_name}")

    # 1. 加载模型
    model, processor = FastVisionModel.from_pretrained(
        "google/gemma-4-E4B-IT",
        load_in_4bit=True,
        use_gradient_checkpointing=True,
    )

    # 2. 应用 PEFT 方法
    if args.peft == "lora":
        model = FastVisionModel.get_peft_model(
            model, r=args.rank, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","proj"]
        )
    elif args.peft == "dora":
        model = FastVisionModel.get_peft_model(
            model, r=args.rank, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","proj"],
            use_dora=True
        )
    elif args.peft == "pissa":
        model = FastVisionModel.get_peft_model(
            model, r=args.rank, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","proj"],
            init_lora_weights="pissa"
        )

    model.print_trainable_parameters()

    # 3. 加载数据集
    if args.dataset == "VQA-RAD":
        dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")
    else:
        dataset = load_dataset("mdwiratathya/SLAKE-vqa-english", split="train")

    # 4. 训练
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            max_steps=200,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=str(output_dir),
            report_to="wandb",
        ),
        data_collator=processor,
    )

    print("开始训练...")
    trainer.train()

    model.save_pretrained(str(output_dir / "final_model"))
    processor.save_pretrained(str(output_dir / "final_model"))

    wandb.finish()
    print(f"✅ 实验完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    main()