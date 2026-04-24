import argparse
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_from_disk


DATASET_ALIASES = {
    "VQA-RAD": "vqa-rad",
    "SLAKE-VQA": "slake",
    "all": "all",
}


def find_first_existing(columns: List[str], candidates: List[str], required: bool = True):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"未找到列，候选={candidates}，实际列={columns}")
    return None


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float, bool)):
        return str(x).strip()
    if isinstance(x, list):
        vals = [normalize_text(v) for v in x]
        vals = [v for v in vals if v]
        return " | ".join(vals)
    if isinstance(x, dict):
        for k in ["text", "answer", "label", "value"]:
            if k in x:
                return normalize_text(x[k])
        return str(x).strip()
    return str(x).strip()


def build_user_prompt(question: str) -> str:
    return (
        "Answer the medical question based on the image. "
        "Be concise and accurate.\n"
        f"Question: {question}"
    )


def resolve_source_columns(columns: List[str]) -> tuple[str, str, str]:
    # Prefer strict mapping when dataset schema is fixed.
    if all(col in columns for col in ["image", "question", "answer"]):
        return "image", "question", "answer"

    image_col = find_first_existing(columns, ["image", "img", "image_file", "image_path", "path"])
    question_col = find_first_existing(columns, ["question", "query", "prompt"])
    answer_col = find_first_existing(columns, ["answer", "answers", "label", "response"])
    return image_col, question_col, answer_col


def convert_split(input_dir: Path, output_dir: Path, dataset_name: str, split: str):
    ds = load_from_disk(str(input_dir))
    columns = ds.column_names

    image_col, question_col, answer_col = resolve_source_columns(columns)
    print(
        f"列映射: image->{image_col}, question->{question_col}, answer->{answer_col}"
    )

    def _map_fn(example: Dict[str, Any]):
        question = normalize_text(example.get(question_col))
        answer = normalize_text(example.get(answer_col))

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": build_user_prompt(question),
                        },
                        {
                            "type": "image",
                            "image": example[image_col],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer,
                        }
                    ],
                },
            ],
        }

    converted = ds.map(
        _map_fn,
        remove_columns=columns,
        desc=f"Converting {dataset_name}/{split}",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    converted.save_to_disk(str(output_dir))
    print(f"✅ 已保存: {output_dir} | 样本数: {len(converted)}")
    print(f"字段: {converted.column_names}")


def main():
    parser = argparse.ArgumentParser(description="极简版 Med-VQA 转视觉对话格式，仅保留 messages")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["VQA-RAD", "SLAKE-VQA", "all"],
        default="all",
        help="要处理的数据集",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="data",
        help="原始本地数据根目录",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/processed",
        help="转换后数据根目录",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    targets = ["VQA-RAD", "SLAKE-VQA"] if args.dataset == "all" else [args.dataset]

    for name in targets:
        folder = DATASET_ALIASES[name]
        dataset_root = input_root / folder
        if not dataset_root.exists():
            raise FileNotFoundError(f"找不到数据目录: {dataset_root}")

        for split in ["train", "validation", "test"]:
            split_dir = dataset_root / split
            if not split_dir.exists():
                print(f"⚠️ 跳过不存在的 split: {split_dir}")
                continue

            out_dir = output_root / folder / split
            convert_split(
                input_dir=split_dir,
                output_dir=out_dir,
                dataset_name=name,
                split=split,
            )


if __name__ == "__main__":
    main()