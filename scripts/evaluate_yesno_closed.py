"""Evaluate closed-set yes/no accuracy on processed MedVQA datasets."""
import unsloth
import argparse
import io
import json
import re
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from PIL import Image
from unsloth import FastVisionModel


DATASET_ALIASES = {
    "VQA-RAD": "vqa-rad",
    "SLAKE-VQA": "slake",
}

YES_WORDS = {"yes"}
NO_WORDS = {"no"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate yes/no closed-set accuracy on processed MedVQA data."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path for inference. Can be base model or trained adapter directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["VQA-RAD", "SLAKE-VQA"],
        required=True,
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/processed",
        help="Root path of processed datasets.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to run.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Maximum number of generated tokens per sample.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save summary and per-sample predictions.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample predictions.",
    )
    parser.add_argument(
        "--preview_first_n",
        type=int,
        default=5,
        help="Print input/output preview for first N evaluated yes/no samples.",
    )
    parser.add_argument(
        "--load_in_4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Load model in 4-bit mode.",
    )
    parser.add_argument(
        "--no_load_in_4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading.",
    )
    parser.set_defaults(load_in_4bit=False)
    return parser.parse_args()

def resolve_data_path(data_root: Path, dataset_name: str, split: str) -> Path:
    folder = DATASET_ALIASES[dataset_name]
    data_path = data_root / folder / split
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset split not found: {data_path}")
    return data_path


def decode_image(raw_image: Any) -> Image.Image:
    if isinstance(raw_image, Image.Image):
        return raw_image.convert("RGB")

    if isinstance(raw_image, dict):
        if raw_image.get("bytes") is not None:
            return Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
        if raw_image.get("path"):
            return Image.open(raw_image["path"]).convert("RGB")

    if isinstance(raw_image, str):
        return Image.open(raw_image).convert("RGB")

    raise TypeError(f"Unsupported image object type: {type(raw_image)}")


def tokenize_text(text: str) -> list[str]:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", str(text).lower()).strip()
    return normalized.split()


def normalize_yes_no(text: str) -> str | None:
    tokens = tokenize_text(text)

    for token in tokens:
        if token in YES_WORDS:
            return "yes"
        if token in NO_WORDS:
            return "no"

    return None


def extract_sample_io(sample: dict[str, Any]) -> tuple[str, Image.Image, str]:
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Sample 'messages' must be a list")

    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)

    if user_msg is None or assistant_msg is None:
        raise ValueError("Sample must contain both user and assistant messages")

    question = "\n".join(
        item.get("text", "")
        for item in user_msg.get("content", [])
        if item.get("type") == "text"
    ).strip()

    if not question:
        raise ValueError("User question text is empty")

    image_item = next(
        (item for item in user_msg.get("content", []) if item.get("type") == "image"),
        None,
    )
    if image_item is None:
        raise ValueError("User message does not contain an image")

    image = decode_image(image_item.get("image"))

    gold_answer = "\n".join(
        item.get("text", "")
        for item in assistant_msg.get("content", [])
        if item.get("type") == "text"
    ).strip()

    return question, image, gold_answer


def build_user_message(question: str) -> list[dict[str, Any]]:
    question = f"{question}\nAnswer with only one word: yes or no."
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]


def main() -> None:
    args = parse_args()
    if args.preview_first_n < 0:
        raise ValueError("--preview_first_n must be >= 0")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3.5-9B inference in this script.")

    data_path = resolve_data_path(Path(args.data_root), args.dataset, args.split)
    dataset = load_from_disk(str(data_path))

    if args.max_samples is not None:
        max_samples = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    print("=" * 88)
    print(f"model_path: {args.model_path}")
    print(f"data_path: {data_path}")
    print(f"num_samples_loaded: {len(dataset)}")
    print(f"load_in_4bit: {args.load_in_4bit}")
    print("=" * 88)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_path,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)

    correct = 0
    closed_total = 0
    skipped_non_binary = 0
    unparseable_pred = 0
    failed_samples = 0
    preview_printed = 0

    records: list[dict[str, Any]] = []

    for idx, sample in enumerate(dataset):
        sample_id = sample.get("id", idx)

        try:
            question, image, gold_raw = extract_sample_io(sample)
        except Exception as e:
            failed_samples += 1
            records.append(
                {
                    "index": idx,
                    "id": sample_id,
                    "error": f"extract_error: {e}",
                }
            )
            continue

        gold = normalize_yes_no(gold_raw)
        if gold is None:
            skipped_non_binary += 1
            records.append(
                {
                    "index": idx,
                    "id": sample_id,
                    "gold_raw": gold_raw,
                    "skipped": "gold_not_yes_no",
                }
            )
            continue

        closed_total += 1

        messages = build_user_message(question)
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(
            text=input_text,
            images=image,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                do_sample=False,
            )

        input_len = int(inputs["input_ids"].shape[1])
        pred_raw = tokenizer.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
        )[0].strip()

        pred = normalize_yes_no(pred_raw)
        if pred is None:
            unparseable_pred += 1

        is_correct = pred == gold
        correct += int(is_correct)

        record = {
            "index": idx,
            "id": sample_id,
            "gold_raw": gold_raw,
            "gold_norm": gold,
            "pred_raw": pred_raw,
            "pred_norm": pred,
            "correct": is_correct,
        }
        records.append(record)

        if preview_printed < args.preview_first_n:
            print("\n" + "-" * 88)
            print(f"input_raw: {question}")
            print(f"output_raw: {pred_raw}")
            print("-" * 88)
            preview_printed += 1

        if args.verbose:
            print(record)
        elif (idx + 1) % 25 == 0:
            print(
                f"processed={idx + 1} closed_total={closed_total} "
                f"correct={correct} running_acc={correct / max(closed_total, 1):.4f}"
            )

    accuracy = correct / closed_total if closed_total > 0 else 0.0

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "model_path": args.model_path,
        "samples_loaded": len(dataset),
        "closed_yes_no_evaluated": closed_total,
        "correct": correct,
        "accuracy": accuracy,
        "skipped_non_binary_gold": skipped_non_binary,
        "unparseable_predictions": unparseable_pred,
        "failed_samples": failed_samples,
    }

    print("\n" + "=" * 88)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 88)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "records": records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
