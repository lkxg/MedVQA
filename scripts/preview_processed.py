# scripts/preview_processed.py
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_from_disk


DATASET_ALIASES = {
    "VQA-RAD": "vqa-rad",
    "SLAKE-VQA": "slake",
}


def image_info(image_obj: Any) -> Dict[str, Any]:
    info = {
        "python_type": str(type(image_obj)),
    }

    if image_obj is None:
        info["is_none"] = True
        return info

    try:
        if hasattr(image_obj, "size"):
            info["size"] = image_obj.size
        if hasattr(image_obj, "mode"):
            info["mode"] = image_obj.mode
        if hasattr(image_obj, "filename"):
            info["filename"] = image_obj.filename
    except Exception as e:
        info["inspect_error"] = str(e)

    if isinstance(image_obj, dict):
        info["keys"] = list(image_obj.keys())
        if "path" in image_obj:
            info["path"] = image_obj["path"]
        if "bytes" in image_obj:
            b = image_obj["bytes"]
            info["bytes_len"] = len(b) if b is not None else None

    return info


def sanitize_content_item(item: Dict[str, Any]) -> Dict[str, Any]:
    item_type = item.get("type", "unknown")

    if item_type == "image":
        image_obj = item.get("image", None)
        return {
            "type": "image",
            "image": "<IMAGE_OBJECT>",
            "image_info": image_info(image_obj),
        }

    if item_type == "text":
        return {
            "type": "text",
            "text": item.get("text", ""),
        }

    safe_item = {"type": item_type}
    for k, v in item.items():
        if k == "image":
            safe_item["image"] = "<IMAGE_OBJECT>" if v is not None else None
        else:
            safe_item[k] = v
    return safe_item


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    safe_messages = []
    for msg in messages:
        safe_messages.append({
            "role": msg.get("role", ""),
            "content": [sanitize_content_item(x) for x in msg.get("content", [])],
        })
    return safe_messages


def extract_quick_view(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    user_texts = []
    assistant_texts = []
    image_count = 0

    for msg in messages:
        role = msg.get("role", "")
        for item in msg.get("content", []):
            if item.get("type") == "image":
                image_count += 1
            elif item.get("type") == "text":
                text = item.get("text", "")
                if role == "user":
                    user_texts.append(text)
                elif role == "assistant":
                    assistant_texts.append(text)

    return {
        "user_text": "\n".join(user_texts).strip(),
        "assistant_text": "\n".join(assistant_texts).strip(),
        "image_count": image_count,
    }


def preview_sample(ds, index: int):
    if index < 0 or index >= len(ds):
        raise IndexError(f"index 越界: {index}, 数据集大小为 {len(ds)}")

    sample = ds[index]
    sample_id = sample.get("id", f"sample-{index}")
    messages = sample.get("messages", [])
    quick = extract_quick_view(messages)

    preview = {
        "dataset_columns": ds.column_names,
        "id": sample_id,
        "quick_view": quick,
        "messages": sanitize_messages(messages),
    }

    print("=" * 100)
    print(f"样本索引: {index}")
    print(json.dumps(preview, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="预览 processed_minimal 后的视觉对话数据")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["VQA-RAD", "SLAKE-VQA"],
        required=True,
        help="数据集名称",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="train",
        help="数据切分",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/processed",
        help="processed 数据根目录",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="查看第几个样本",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="连续查看多少个样本",
    )
    args = parser.parse_args()

    folder = DATASET_ALIASES[args.dataset]
    data_path = Path(args.data_root) / folder / args.split

    if not data_path.exists():
        raise FileNotFoundError(f"找不到 processed 数据目录: {data_path}")

    ds = load_from_disk(str(data_path))

    print(f"数据路径: {data_path}")
    print(f"样本总数: {len(ds)}")
    print(f"字段列表: {ds.column_names}")

    end_index = min(args.index + args.num_samples, len(ds))
    for idx in range(args.index, end_index):
        preview_sample(ds, idx)


if __name__ == "__main__":
    main()