import unsloth
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, set_seed
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# 数据集名称到文件夹名称的映射
DATASET_ALIASES = {
    "VQA-RAD": "vqa-rad",
    "SLAKE-VQA": "slake",
}


def parse_args():
    """解析用于训练 Qwen3.5-9B 的命令行参数。"""
    parser = argparse.ArgumentParser(description="在极简 Med-VQA 对话数据集上训练 Qwen3.5-9B")

    # 路径与数据集配置
    parser.add_argument("--model_path", type=str, default="models/Qwen3.5-9B", help="基础模型目录路径")
    parser.add_argument("--dataset", type=str, choices=["VQA-RAD", "SLAKE-VQA"], required=True, help="目标训练数据集")
    parser.add_argument("--data_root", type=str, default="data/processed", help="预处理后数据的根目录")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen35-9b", help="保存 adapter 模型和日志的目录")

    # PEFT (Parameter-Efficient Fine-Tuning) 参数
    parser.add_argument("--peft", type=str, choices=["lora", "dora", "pissa"], default="lora", help="使用的 PEFT 方法")
    parser.add_argument("--rank", type=int, default=16, help="LoRA 秩 (rank)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha 参数")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout 丢弃率")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=2, help="每个设备的训练批量大小")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="每个设备的评估批量大小")
    parser.add_argument("--grad_accum", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数 (epochs)")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大训练步数（-1 表示使用 epochs）")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="峰值学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="优化器的权重衰减 (weight decay)")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录的步数间隔")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存的检查点 (checkpoints) 最大数量")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="序列的最大上下文长度")
    parser.add_argument("--seed", type=int, default=3407, help="控制可重复性的随机种子")

    # 额外选项
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="启用 4-bit 量化")
    parser.add_argument("--no_load_in_4bit", action="store_true", help="禁用 4-bit 量化")
    parser.add_argument("--run_name", type=str, default=None, help="当前训练运行的自定义名称")

    # 通过 wandb 记录日志
    parser.add_argument("--use_wandb", action="store_true", default=True, help="启用 W&B 日志记录（默认开启）")
    parser.add_argument("--no_wandb", action="store_true", help="禁用 W&B 日志记录")
    parser.add_argument("--wandb_project", type=str, default="MedVQA-Qwen35", help="W&B 项目名称")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B 运行名称")

    return parser.parse_args()


def resolve_paths(data_root: Path, dataset_name: str) -> tuple[Path, Path]:
    """解析并验证训练阶段和验证阶段数据集目录的路径。"""
    folder = DATASET_ALIASES[dataset_name]
    train_path = data_root / folder / "train"
    val_path = data_root / folder / "validation"

    if not train_path.exists():
        raise FileNotFoundError(f"未找到指定的训练数据集，路径: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"未找到指定的验证数据集，路径: {val_path}")

    return train_path, val_path


def validate_message_item(item: Dict[str, Any]) -> None:
    """验证消息内容 (message content) 中的单个子项。"""
    if "type" not in item:
        raise ValueError(f"内容项缺少 'type' 字段: {item}")

    item_type = item["type"]
    if item_type == "text":
        if "text" not in item:
            raise ValueError(f"类型为 text 的内容项缺少 'text' 字段: {item}")
    elif item_type == "image":
        if "image" not in item:
            raise ValueError(f"类型为 image 的内容项缺少 'image' 字段: {item}")


def validate_messages(messages: list[Dict[str, Any]]) -> None:
    """验证视觉-语言数据的对话消息布局（OpenAI 风格）。"""
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"消息结构无效（期望是长度 >= 2 的列表）: {messages}")

    for msg in messages:
        if "role" not in msg or "content" not in msg:
            raise ValueError(f"消息缺少 'role' 或 'content' 字段: {msg}")
        if not isinstance(msg["content"], list):
            raise ValueError(f"消息中的 'content' 必须是列表: {msg}")
        for item in msg["content"]:
            validate_message_item(item)

    has_user = any(m.get("role") == "user" for m in messages)
    has_assistant = any(m.get("role") == "assistant" for m in messages)

    if not has_user or not has_assistant:
        raise ValueError(f"消息中必须包含 'user' 和 'assistant' 角色: {messages}")


def validate_dataset(ds, name: str) -> None:
    """验证加载的数据集包含必需的列并且各个项目的格式正确。"""
    required_columns = {"messages"}
    actual_columns = set(ds.column_names)

    if required_columns - actual_columns:
        raise ValueError(
            f"数据集 {name} 缺少必需的字段: {required_columns - actual_columns}。当前的字段是: {ds.column_names}"
        )

    # 快速验证第一条样本的数据格式
    sample = ds[0]
    validate_messages(sample["messages"])


def preview_sample(ds, title: str) -> None:
    """安全地显示一条样本预览，避免将繁重的原始图像对象打印到标准输出导致的错误。"""
    sample = ds[0]
    safe_preview = {"messages": []}
    if "id" in sample:
        safe_preview["id"] = sample.get("id")

    for msg in sample["messages"]:
        safe_msg = {"role": msg["role"], "content": []}
        for item in msg["content"]:
            if item["type"] == "image":
                safe_msg["content"].append({
                    "type": "image",
                    "image": "<IMAGE_OBJECT>",  # 屏蔽原始图像显示
                })
            elif item["type"] == "text":
                safe_msg["content"].append({
                    "type": "text",
                    "text": item.get("text", ""),
                })
            else:
                safe_msg["content"].append(item)
        safe_preview["messages"].append(safe_msg)

    logger.info(f"===== {title} 样本预览 =====\n"
                f"{json.dumps(safe_preview, ensure_ascii=False, indent=2)}\n"
                f"===========================")


def apply_peft(model, args):
    """根据选择使用 LoRA、DoRA 或 PiSSA 方法，将 PEFT 配置应用于模型。"""
    kwargs = dict(
        # 微调具体的多个投射模块，包括视觉映射模块等
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        # target_modules="all-linear",
    )

    if args.peft == "lora":
        model = FastVisionModel.get_peft_model(model, **kwargs)
    elif args.peft == "dora":
        model = FastVisionModel.get_peft_model(model, use_dora=True, **kwargs)
    elif args.peft == "pissa":
        model = FastVisionModel.get_peft_model(model, init_lora_weights="pissa", **kwargs)
    else:
        raise ValueError(f"不支持的 PEFT 机制（方法）: {args.peft}")

    return model


def maybe_init_wandb(args, run_name: str) -> str:
    """按需初始化 W&B 追踪，返回为 TRL `report_to` 所需的名称。"""
    if not args.use_wandb:
        return "none"

    if not WANDB_AVAILABLE:
        logger.warning("未安装 wandb 模块。将不会追踪 W&B 进度记录。")
        return "none"

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or run_name,
        config=vars(args),
    )
    return "wandb"


def main():
    args = parse_args()

    # 设置全局随机种子确保可重复性
    set_seed(args.seed)

    # 如果有明确指定，则无条件禁用 4-bit 模型加载重定向
    if args.no_load_in_4bit:
        args.load_in_4bit = False
        
    if args.no_wandb:
        args.use_wandb = False

    # 设置保存相关的目录
    run_name = args.run_name or f"qwen35-9b-{args.dataset.lower()}-{args.peft}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("解析路径并开始加载相关数据集...")
    train_path, val_path = resolve_paths(Path(args.data_root), args.dataset)
    train_dataset = load_from_disk(str(train_path))
    val_dataset = load_from_disk(str(val_path))

    # 执行数据集结构格式的全面验证
    validate_dataset(train_dataset, "train_dataset")
    validate_dataset(val_dataset, "val_dataset")
    preview_sample(train_dataset, "TRAIN")

    logger.info(f"🚀 运行名称 (Run name): {run_name}")
    logger.info(f"📦 模型路径 (Model path): {args.model_path}")
    logger.info(f"🧠 训练路径 (Train path): {train_path} (样本数: {len(train_dataset)})")
    logger.info(f"🧪 验证路径 (Val path):   {val_path} (样本数: {len(val_dataset)})")
    logger.info(f"💾 输出目录 (Output dir): {output_dir}")

    report_to = maybe_init_wandb(args, run_name)

    logger.info("基于 Unsloth 框架初始化 FastVisionModel 模型...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_path,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_length,
    )

    logger.info("将选择的 PEFT adapter 应用到模型中...")
    model = apply_peft(model, args)
    FastVisionModel.for_training(model)

    try:
        model.print_trainable_parameters()  # 打印可被训练的参数占用百分比
    except Exception:
        pass

    data_collator = UnslothVisionDataCollator(model, tokenizer)

    logger.info("准备相关的 SFTTrainer 及其内置配置...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=15,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=str(output_dir),
            report_to=report_to,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=25,
            eval_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=args.save_total_limit,

            # Unsloth 需要的特定配置修改覆盖
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_seq_length,
        ),
    )

    logger.info("正式开始启动训练阶段轮询...")
    train_result = trainer.train()

    logger.info("保存指定的 LoRA/PEFT adapters 以及重写 Tokenizer配置...")
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    logger.info("持久化保存 SFTTrainer 在此阶段内部的所有相关基础状态...")
    trainer.save_state()

    metrics = train_result.metrics
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 并行训练阶段全流程任务圆满完成! 适配器保存记录路径: {adapter_dir}")
    logger.info(f"📊 完整模型训练分析记录相关指标被正式序列并封存在: {metrics_path}")

    if report_to == "wandb" and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()