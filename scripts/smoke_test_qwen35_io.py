#!/usr/bin/env python3
"""Smoke test for Qwen3.5 vision model input/output on processed MedVQA data."""

import argparse
import io
import json
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401  # Required by Unsloth runtime patching side effects.
import torch
from datasets import load_from_disk
from PIL import Image
from unsloth import FastVisionModel


DATASET_ALIASES = {
    "VQA-RAD": "vqa-rad",
    "SLAKE-VQA": "slake",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-sample Qwen3.5 vision smoke test and print input/output details."
    )
    parser.add_argument("--model_path", type=str, default="models/Qwen3.5-9B", help="Model or adapter directory")
    parser.add_argument("--dataset", type=str, choices=["VQA-RAD", "SLAKE-VQA"], required=True)
    parser.add_argument("--split", type=str, choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--data_root", type=str, default="data/processed", help="Root of processed datasets")
    parser.add_argument("--index", type=int, default=0, help="Sample index in selected split")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--question_override", type=str, default=None, help="Replace sample question with custom text")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Enable 4-bit loading (default: true)")
    parser.add_argument("--no_load_in_4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--temperature", type=float, default=0.0, help="Set > 0 for sampling")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--skip_generate", action="store_true", help="Only inspect inputs and skip generation")
    parser.add_argument("--template_preview_chars", type=int, default=1200, help="Max chars shown for chat template")
    parser.add_argument("--save_image_path", type=str, default=None, help="Optional path to save decoded sample image")
    return parser.parse_args()


def resolve_data_path(data_root: Path, dataset_name: str, split: str) -> Path:
    folder = DATASET_ALIASES[dataset_name]
    data_path = data_root / folder / split
    if not data_path.exists():
        raise FileNotFoundError(f"Processed split not found: {data_path}")
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


def extract_sample_parts(sample: dict[str, Any]) -> tuple[str, str, str, Image.Image, list[dict[str, Any]]]:
    sample_id = str(sample.get("id", "unknown"))
    messages = sample.get("messages", [])

    if not isinstance(messages, list) or not messages:
        raise ValueError("Sample has no valid 'messages' list")

    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)

    if user_msg is None or assistant_msg is None:
        raise ValueError("Sample messages must include both 'user' and 'assistant' roles")

    question = "\n".join(
        item.get("text", "")
        for item in user_msg.get("content", [])
        if item.get("type") == "text"
    ).strip()

    reference_answer = "\n".join(
        item.get("text", "")
        for item in assistant_msg.get("content", [])
        if item.get("type") == "text"
    ).strip()

    image_item = next((item for item in user_msg.get("content", []) if item.get("type") == "image"), None)
    if image_item is None:
        raise ValueError("No user image found in sample")

    image = decode_image(image_item.get("image"))

    return sample_id, question, reference_answer, image, messages


def summarize_message_structure(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for msg in messages:
        summary.append(
            {
                "role": msg.get("role", ""),
                "content_types": [item.get("type", "unknown") for item in msg.get("content", [])],
            }
        )
    return summary


def main() -> None:
    args = parse_args()

    if args.no_load_in_4bit:
        args.load_in_4bit = False

    data_path = resolve_data_path(Path(args.data_root), args.dataset, args.split)
    ds = load_from_disk(str(data_path))

    if args.index < 0 or args.index >= len(ds):
        raise IndexError(f"index out of range: {args.index}, dataset size: {len(ds)}")

    sample = ds[args.index]
    sample_id, question, reference_answer, image, messages = extract_sample_parts(sample)

    if args.question_override:
        question = args.question_override

    if not question:
        question = "Describe the key findings in this medical image."

    if args.save_image_path:
        image.save(args.save_image_path)

    print("=" * 88)
    print("[1] Dataset Sample")
    print(f"data_path: {data_path}")
    print(f"num_samples: {len(ds)}")
    print(f"sample_index: {args.index}")
    print(f"sample_id: {sample_id}")
    print(f"image_size: {image.size}, image_mode: {image.mode}")
    print("message_structure:")
    print(json.dumps(summarize_message_structure(messages), indent=2, ensure_ascii=False))
    print("\nquestion:")
    print(question)
    print("\nreference_answer:")
    print(reference_answer if reference_answer else "<EMPTY>")

    print("\n[2] Loading Model")
    print(f"model_path: {args.model_path}")
    print(f"load_in_4bit: {args.load_in_4bit}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_path,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_length,
    )
    FastVisionModel.for_inference(model)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    input_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    preview_chars = max(120, args.template_preview_chars)
    shown_text = input_text[:preview_chars]
    if len(input_text) > preview_chars:
        shown_text += "\n...<truncated>"

    print("\n[3] Prompt / Tokenizer Input")
    print(shown_text)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\ncompute_device: {device}")

    inputs = tokenizer(
        text=input_text,
        images=image,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    print("tensor_keys:", list(inputs.keys()))
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")

    if args.skip_generate:
        print("\n[4] Generation skipped (--skip_generate).")
        print("=" * 88)
        return

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }

    if args.temperature > 0:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        })
    else:
        generation_kwargs.update({"do_sample": False})

    print("\n[4] Generating Output")
    print("generation_config:", generation_kwargs)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    input_token_count = int(inputs["input_ids"].shape[1])
    pred_text = tokenizer.batch_decode(
        outputs[:, input_token_count:],
        skip_special_tokens=True,
    )[0].strip()

    print("\nmodel_output:")
    print(pred_text if pred_text else "<EMPTY_OUTPUT>")
    print("=" * 88)


if __name__ == "__main__":
    main()
