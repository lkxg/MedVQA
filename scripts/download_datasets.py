import os
from datasets import load_dataset

def download_and_save_dataset(repo_id, save_folder):
    """
    下载 HuggingFace 数据集并保存到本地
    """
    print(f"🚀 正在连接 Hugging Face 获取: {repo_id}...")
    try:
        # load_dataset 会自动处理文本和图片数据的下载与对齐
        dataset = load_dataset(repo_id)
        
        # 保存到本地磁盘
        dataset.save_to_disk(save_folder)
        print(f"✅ {repo_id} 成功下载并保存至: {save_folder}\n")
        
        # 打印一下数据集的结构和大小
        print("📊 数据集结构概览:")
        print(dataset)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ 下载 {repo_id} 时发生错误: {e}")

if __name__ == "__main__":
    # 如果你在国内网络环境，强烈建议在代码里开启国内镜像源加速
    #os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 创建本地 data 目录
    BASE_DATA_DIR = "./data"
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    
    # 1. 下载 VQA-RAD 数据集
    vqa_rad_repo = "flaviagiammarino/vqa-rad"
    vqa_rad_save_path = os.path.join(BASE_DATA_DIR, "vqa-rad")
    download_and_save_dataset(vqa_rad_repo, vqa_rad_save_path)
    
    # 2. 下载 SLAKE-VQA (英文版) 数据集
    slake_repo = "mdwiratathya/SLAKE-vqa-english"
    slake_save_path = os.path.join(BASE_DATA_DIR, "slake")
    download_and_save_dataset(slake_repo, slake_save_path)
    
    print("🎉 所有数据集下载任务已完成！")