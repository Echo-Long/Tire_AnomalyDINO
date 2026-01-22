import os
import requests
from tqdm import tqdm

# 配置信息
DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
TARGET_PATH = "/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"

def download_dinov2_weights(url, target_path):
    # 创建父目录（如果不存在）
    parent_dir = os.path.dirname(target_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"已创建目录: {parent_dir}")
    
    # 检查文件是否已存在
    if os.path.exists(target_path):
        file_size = os.path.getsize(target_path) / (1024 * 1024)  # 转换为MB
        print(f"文件已存在: {target_path} (大小: {file_size:.2f}MB)")
        # 简单校验文件大小（官方小模型约84MB）
        if 80 < file_size < 90:
            print("文件大小正常，无需重复下载")
            return
        else:
            print("文件大小异常，将重新下载...")
    
    try:
        # 发送请求（流式下载，支持大文件）
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # 抛出HTTP错误
        
        # 获取文件总大小
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024  # 1MB/块
        
        # 显示下载进度条
        print(f"开始下载: {url}")
        with open(target_path, "wb") as file, tqdm(
            desc="下载进度",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        
        # 下载完成后校验
        final_size = os.path.getsize(target_path) / (1024 * 1024)
        if 80 < final_size < 90:
            print(f"\n下载成功！文件保存至: {target_path}")
            print(f"最终文件大小: {final_size:.2f}MB")
        else:
            print(f"\n警告：文件大小异常 ({final_size:.2f}MB)，可能下载不完整！")
    
    except requests.exceptions.RequestException as e:
        print(f"\n下载失败: {str(e)}")
        # 清理不完整的文件
        if os.path.exists(target_path):
            os.remove(target_path)
            print("已删除不完整文件")
    except Exception as e:
        print(f"\n未知错误: {str(e)}")

if __name__ == "__main__":
    download_dinov2_weights(DOWNLOAD_URL, TARGET_PATH)