
"""
download_models.py
MedVQA-PEFT 项目专用模型下载脚本
功能：一键下载 Qwen3.5-9B 和 Gemma-4-E4B-IT 模型，支持断点续传和进度显示。
"""
import argparse
import os
from huggingface_hub import snapshot_download, login
from pathlib import Path

# ====================== 配置区 ======================
MODEL_CONFIG = {
    "qwen3.5-9b": {
        "repo_id": "Qwen/Qwen3.5-9B",
        "save_dirname": "Qwen3.5-9B",
        "description": "Qwen3.5-9B (9B 原生多模态)",
    },
    "gemma-4-e4b": {
        "repo_id": "google/gemma-4-E4B-it",
        "save_dirname": "gemma-4-E4B-it",
        "description": "Gemma-4-E4B-IT (边缘优化多模态)",
    },
}

DEFAULT_OUTPUT_DIR = "./models"   # 默认保存路径


def download_model(repo_id: str, save_dir: str, token: str = None):
    """下载模型（已移除不兼容参数）"""
    print(f"🚀 开始下载模型: {repo_id}")
    print(f"   保存路径: {save_dir}\n")

    if token:
        login(token=token)
        print("✅ Hugging Face Token 登录成功\n")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=save_dir,
        )
        print(f"✅ 模型 {repo_id} 下载完成！\n")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="MedVQA-PEFT 项目模型下载工具（修复版）")
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen3.5-9b", "gemma-4-e4b", "all"],
        default="all",
        help="要下载的模型 (默认: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"模型保存目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face Access Token（Qwen系列通常需要）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅列出支持的模型",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("📋 支持下载的模型：")
        for name, info in MODEL_CONFIG.items():
            print(f"   • {name:<12} → {info['description']}")
        return

    models_to_download = list(MODEL_CONFIG.keys()) if args.model == "all" else [args.model]

    print(f"📥 即将下载 {len(models_to_download)} 个模型...\n")

    for model_key in models_to_download:
        config = MODEL_CONFIG[model_key]
        save_dir = output_path / config["save_dirname"]
        download_model(config["repo_id"], str(save_dir), args.hf_token)

    print("🎉 所有模型下载完成！")
    print(f"📁 保存位置: {output_path.absolute()}")


if __name__ == "__main__":
    main()