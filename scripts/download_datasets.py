import os
from datasets import DatasetDict, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download

def download_and_save_dataset(repo_id, save_folder, split=None):
    """
    下载 HuggingFace 数据集并保存到本地
    """
    split_info = f" (split={split})" if split else ""
    print(f"🚀 正在连接 Hugging Face 获取: {repo_id}{split_info}...")
    try:
        # load_dataset 会自动处理文本和图片数据的下载与对齐
        if split:
            dataset = load_dataset(repo_id, split=split)
        else:
            dataset = load_dataset(repo_id)
        
        # 保存到本地磁盘
        dataset.save_to_disk(save_folder)
        print(f"✅ {repo_id}{split_info} 成功下载并保存至: {save_folder}\n")
        
        # 打印一下数据集的结构和大小
        print("📊 数据集结构概览:")
        print(dataset)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ 下载 {repo_id} 时发生错误: {e}")

def download_pmc_vqa_test_clean(save_folder):
    """
    专门下载 RadGenome/PMC-VQA 的 test_clean.csv。
    该仓库不同 CSV 列名不一致，不能直接走默认 load_dataset(repo_id) 构建流程。
    """
    print("🚀 正在连接 Hugging Face 获取: RadGenome/PMC-VQA (test_clean.csv)...")
    try:
        local_csv_path = hf_hub_download(
            repo_id="RadGenome/PMC-VQA",
            repo_type="dataset",
            filename="test_clean.csv",
        )

        dataset = load_dataset("csv", data_files=local_csv_path, split="train")
        dataset.save_to_disk(save_folder)

        print(f"✅ RadGenome/PMC-VQA (test_clean.csv) 成功下载并保存至: {save_folder}\n")
        print("📊 数据集结构概览:")
        print(dataset)
        print("-" * 50)
    except Exception as e:
        print(f"❌ 下载 RadGenome/PMC-VQA 的 test_clean.csv 时发生错误: {e}")


def create_vqa_rad_validation_split(
    source_folder,
    output_folder,
    val_ratio=0.20,
    seed=42,
):
    """
    从 VQA-RAD 的 train split 中按固定随机种子切分验证集。
    输出为包含 train/validation/test 的 DatasetDict。
    """
    print(
        f"🚀 正在生成 VQA-RAD 验证集切分: val_ratio={val_ratio}, seed={seed}..."
    )

    try:
        dataset_dict = load_from_disk(source_folder)
        if "train" not in dataset_dict or "test" not in dataset_dict:
            raise ValueError("VQA-RAD 数据缺少 train 或 test split，无法自动切分。")

        split_result = dataset_dict["train"].train_test_split(
            test_size=val_ratio,
            seed=seed,
        )

        dataset_with_val = DatasetDict(
            {
                "train": split_result["train"],
                "validation": split_result["test"],
                "test": dataset_dict["test"],
            }
        )

        dataset_with_val.save_to_disk(output_folder)

        print(f"✅ VQA-RAD 切分完成并保存至: {output_folder}")
        print(
            "📊 切分后样本数: "
            f"train={len(dataset_with_val['train'])}, "
            f"validation={len(dataset_with_val['validation'])}, "
            f"test={len(dataset_with_val['test'])}"
        )
        print("-" * 50)
    except Exception as e:
        print(f"❌ 生成 VQA-RAD 验证集切分时发生错误: {e}")

if __name__ == "__main__":
    # 如果你在国内网络环境，强烈建议在代码里开启国内镜像源加速
    #os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 创建本地 data 目录
    BASE_DATA_DIR = "./data"
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    
    datasets_to_download = [
        {
            "repo_id": "flaviagiammarino/vqa-rad",
            "save_path": os.path.join(BASE_DATA_DIR, "vqa-rad"),
            "split": None,
        },
        {
            "repo_id": "mdwiratathya/SLAKE-vqa-english",
            "save_path": os.path.join(BASE_DATA_DIR, "slake"),
            "split": None,
        },
    ]

    for item in datasets_to_download:
        download_and_save_dataset(
            repo_id=item["repo_id"],
            save_folder=item["save_path"],
            split=item["split"],
        )

    # 为 VQA-RAD 额外生成 train/validation/test 三分割版本。
    create_vqa_rad_validation_split(
        source_folder=os.path.join(BASE_DATA_DIR, "vqa-rad"),
        output_folder=os.path.join(BASE_DATA_DIR, "vqa-rad-split"),
        val_ratio=0.20,
        seed=42,
    )

    # 单独处理 PMC-VQA：仅下载 test_clean.csv（对应 README 中的 test-clean 子集）
    download_pmc_vqa_test_clean(
        save_folder=os.path.join(BASE_DATA_DIR, "pmc-vqa-test-clean")
    )
    
    print("🎉 所有数据集下载任务已完成！")
