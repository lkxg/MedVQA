# scripts/preview_processed.py
import argparse
import io
import json
from pathlib import Path

from datasets import load_from_disk
from PIL import Image


DATASET_ALIASES = {
    "VQA-RAD": "vqa-rad",
    "SLAKE-VQA": "slake",
}


def list_available_processed(data_root: Path) -> None:
    print(f"processed 根目录: {data_root}")
    if not data_root.exists():
        print("目录不存在")
        return

    found_any = False
    for dataset_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        splits = [p.name for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
        if splits:
            found_any = True
            print(f"- {dataset_dir.name}: {', '.join(splits)}")

    if not found_any:
        print("未发现可用的数据集 split 目录")


def resolve_data_path(
    data_root: Path,
    dataset: str | None,
    split: str,
    data_path_arg: str | None,
) -> Path:
    if data_path_arg:
        data_path = Path(data_path_arg)
    else:
        if dataset is None:
            raise ValueError("未指定 --data_path 时，必须提供 --dataset")
        folder = DATASET_ALIASES[dataset]
        data_path = data_root / folder / split

    if not data_path.exists():
        raise FileNotFoundError(f"找不到 processed 数据目录: {data_path}")

    return data_path


def preview_sample(ds, index: int):
    if index < 0 or index >= len(ds):
        raise IndexError(f"index 越界: {index}, 数据集大小为 {len(ds)}")

    sample = ds[index]
    messages = []
    for msg in sample.get("messages", []):
        safe_msg = {
            "role": msg.get("role"),
            "content": [],
        }
        for item in msg.get("content", []):
            if item.get("type") == "image":
                raw_image = item.get("image")
                image_value = "<IMAGE_OBJECT>"

                if isinstance(raw_image, dict):
                    if raw_image.get("bytes") is not None:
                        try:
                            img = Image.open(io.BytesIO(raw_image["bytes"]))
                            image_value = f"<PIL.Image mode={img.mode} size={img.size}>"
                        except Exception:
                            image_value = "<IMAGE_BYTES>"
                    elif raw_image.get("path"):
                        image_value = f"<IMAGE_PATH:{raw_image['path']}>"
                elif raw_image is not None:
                    image_value = str(raw_image)

                safe_msg["content"].append({
                    "type": "image",
                    "image": image_value,
                })
            else:
                if item.get("type") == "text":
                    safe_msg["content"].append({
                        "type": "text",
                        "text": item.get("text", ""),
                    })
                else:
                    safe_msg["content"].append(item)
        messages.append(safe_msg)

    print(json.dumps({"messages": messages}, ensure_ascii=False, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="预览 processed 后的视觉对话数据")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["VQA-RAD", "SLAKE-VQA"],
        default=None,
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
        "--data_path",
        type=str,
        default=None,
        help="直接指定要预览的 split 路径，例如 data/processed/vqa-rad/train",
    )
    parser.add_argument(
        "--list_available",
        action="store_true",
        help="列出 data_root 下可用的数据集和 split",
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

    data_root = Path(args.data_root)
    if args.list_available:
        list_available_processed(data_root)
        return

    data_path = resolve_data_path(
        data_root=data_root,
        dataset=args.dataset,
        split=args.split,
        data_path_arg=args.data_path,
    )

    ds = load_from_disk(str(data_path))

    end_index = min(args.index + args.num_samples, len(ds))
    for idx in range(args.index, end_index):
        preview_sample(ds, idx)


if __name__ == "__main__":
    main()