import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

# -------------------------- 配置参数 --------------------------
RAW_DATA_DIR = Path("/root/autodl-tmp/Tire/PatchCore_Dataset_224x224/tire_block3/train/good")  # 原始800张图片的目录
PROCESSED_DIR = Path("/root/autodl-tmp/Tire/AnomalyDINO_Dataset/train/good")  # 处理后保存的目录
TARGET_SIZE = (224, 224)  # 目标尺寸
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")  # 支持的图片格式
DIGIT_LENGTH = 4  # 编号位数（如4位：0001、0002...）

# -------------------------- 核心函数 --------------------------
def is_black_image(image: Image.Image, threshold: int = 0) -> bool:
    """检测图片是否为全黑（所有像素值≤threshold，默认0）"""
    img_array = np.array(image.convert("L"))  # 转灰度图，简化判断
    return np.all(img_array <= threshold)

def validate_image_size(image: Image.Image, target_size: tuple) -> bool:
    """验证图片尺寸是否符合目标尺寸"""
    return image.size == target_size

def process_dataset():
    # 1. 创建处理后目录（若不存在）
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    # 备份原始数据（可选，建议执行）
    backup_dir = Path("./raw_dataset_backup")
    if not backup_dir.exists():
        shutil.copytree(RAW_DATA_DIR, backup_dir)
        print(f"已备份原始数据至：{backup_dir}")

    # 2. 遍历原始目录，筛选有效图片
    valid_images = []
    black_image_count = 0
    invalid_size_count = 0

    # 按文件名排序（保证编号顺序可追溯）
    raw_image_paths = sorted([p for p in RAW_DATA_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])

    for img_path in raw_image_paths:
        try:
            with Image.open(img_path) as img:
                # 校验尺寸
                if not validate_image_size(img, TARGET_SIZE):
                    invalid_size_count += 1
                    print(f"跳过：{img_path.name} 尺寸不符（实际：{img.size}，目标：{TARGET_SIZE}）")
                    continue
                # 检测全黑
                if is_black_image(img):
                    black_image_count += 1
                    print(f"跳过：{img_path.name} 为全黑图片")
                    continue
                # 有效图片加入列表
                valid_images.append(img_path)
        except Exception as e:
            print(f"跳过：{img_path.name} 读取失败，错误：{e}")

    # 3. 对有效图片统一编号并重命名
    sample_data = []  # 用于生成Anomalib兼容的DataFrame
    for idx, img_path in enumerate(valid_images, start=1):
        # 生成新文件名（如0001.jpg、0002.png）
        new_filename = f"{str(idx).zfill(DIGIT_LENGTH)}{img_path.suffix.lower()}"
        new_path = PROCESSED_DIR / new_filename
        
        # 复制并改名（避免覆盖，也可改用shutil.move直接移动）
        shutil.copy2(img_path, new_path)
        
        # 记录样本信息（适配Anomalib）
        sample_data.append({
            "path": str(PROCESSED_DIR),
            "split": "train",  # 正样本默认划分为训练集
            "label": "good",
            "image_path": str(new_path),
            "mask_path": "",  # 正样本无掩码
            "label_index": 0
        })

    # 4. 生成Anomalib兼容的DataFrame（可选）
    samples_df = pd.DataFrame(sample_data)
    samples_df.attrs["task"] = "classification"  # 仅分类任务（无掩码）
    samples_df.to_csv(PROCESSED_DIR / "samples.csv", index=False)

    # 5. 输出处理日志
    print("\n===== 处理完成 =====")
    print(f"原始图片总数：{len(raw_image_paths)}")
    print(f"全黑图片数量：{black_image_count}")
    print(f"尺寸异常数量：{invalid_size_count}")
    print(f"有效图片数量：{len(valid_images)}")
    print(f"处理后保存至：{PROCESSED_DIR}")
    print(f"Anomalib样本列表已保存：{PROCESSED_DIR / 'samples.csv'}")

# -------------------------- 执行处理 --------------------------
if __name__ == "__main__":
    process_dataset()