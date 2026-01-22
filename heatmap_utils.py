import os
import numpy as np
import torch
from PIL import Image, ImageOps
import json
from pathlib import Path

# -------------------------- 核心优化：真缺陷分数阈值（可根据效果调整） --------------------------
TRUE_DEFECT_SCORE_THRESHOLD = 0.7  # 建议范围0.7~0.9，误判多就调高，缺陷少就调低

# -------------------------- 核心工具：Jet色板映射（替代matplotlib，原始分辨率） --------------------------
def apply_jet_colormap(gray_array):
    """
    将0-1的灰度数组转换为Jet色板的RGB数组（原始分辨率，无缩放）
    :param gray_array: 形状(H, W)，值范围0~1的float数组
    :return: 形状(H, W, 3)，值范围0~255的uint8 RGB数组
    """
    # Jet色板的颜色渐变映射（精准复刻matplotlib的jet）
    def jet_color(value):
        if value < 0.0:
            return (0, 0, 128)
        elif value < 0.125:
            return (0, 0, 255)
        elif value < 0.375:
            return (0, 255, 255)
        elif value < 0.625:
            return (255, 255, 0)
        elif value < 0.875:
            return (255, 0, 0)
        else:
            return (128, 0, 0)
    
    # 向量化计算，避免循环（提升速度）
    h, w = gray_array.shape
    rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 分区间赋值
    mask_0_0125 = (gray_array >= 0.0) & (gray_array < 0.125)
    mask_0125_0375 = (gray_array >= 0.125) & (gray_array < 0.375)
    mask_0375_0625 = (gray_array >= 0.375) & (gray_array < 0.625)
    mask_0625_0875 = (gray_array >= 0.625) & (gray_array < 0.875)
    mask_0875_1 = (gray_array >= 0.875) & (gray_array <= 1.0)
    
    # 蓝色→青
    rgb_array[mask_0_0125] = (0, 0, 255)
    # 青→黄
    rgb_array[mask_0125_0375] = (0, 255, 255)
    # 黄→红
    rgb_array[mask_0375_0625] = (255, 255, 0)
    # 红→深红
    rgb_array[mask_0625_0875] = (255, 0, 0)
    rgb_array[mask_0875_1] = (128, 0, 0)
    
    return rgb_array

# -------------------------- 核心工具：图像叠加（原始分辨率） --------------------------
def overlay_heatmap_on_image(img_array, heatmap_array, alpha=0.6):
    """
    原始分辨率下将热力图叠加到原图上
    :param img_array: 形状(H, W, 3)，0~255 uint8 原图数组
    :param heatmap_array: 形状(H, W, 3)，0~255 uint8 热力图数组
    :param alpha: 热力图透明度
    :return: 形状(H, W, 3)，0~255 uint8 叠加后数组
    """
    # 转换为float避免溢出
    img_float = img_array.astype(np.float32)
    heatmap_float = heatmap_array.astype(np.float32)
    
    # 叠加计算
    overlay_float = (1 - alpha) * img_float + alpha * heatmap_float
    # 裁剪到0-255并转回uint8
    overlay_array = np.clip(overlay_float, 0, 255).astype(np.uint8)
    
    return overlay_array

# -------------------------- DinoV2权重路径修复函数 --------------------------
def fix_dinov2_weight_path():
    from anomalib.models.components.dinov2.dinov2_loader import DinoV2Loader
    def fixed_get_weight_path(self, model_type, architecture, patch_size):
        return Path("/root/autodl-tmp/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    DinoV2Loader._get_weight_path = fixed_get_weight_path
    print("✅ 已强制覆盖DinoV2Loader的权重路径计算逻辑")

# -------------------------- 热力图生成函数（移除第四个子图单独保存） --------------------------
def generate_valid_heatmap(
    image_path, anomaly_map, save_path, global_min, global_max, defect_threshold, 
    image_size, defect_percentile
):
    """
    生成512×512的热力图（仅保存4合1小图，移除单独保存第四个子图逻辑）
    """
    # 读取并处理原图
    raw_image = Image.open(image_path).convert("RGB")
    raw_image = raw_image.resize(image_size)
    raw_image_np = np.array(raw_image)
    
    # 处理异常图
    anomaly_map = anomaly_map.cpu().detach().numpy()
    anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map.shape != image_size:
        anomaly_map = np.resize(anomaly_map, image_size)
    
    # 单张图片的极值
    local_min = anomaly_map.min()
    local_max = anomaly_map.max()
    diff_val = local_max - local_min
    print(f"\n【{os.path.basename(image_path)}】")
    print(f"  单张异常分数范围：{local_min:.6f} ~ {local_max:.6f} | 差值：{diff_val:.6f}")
    print(f"  全局异常分数范围：{global_min:.6f} ~ {global_max:.6f}")
    print(f"  缺陷筛选阈值：{defect_threshold:.6f} | 真缺陷分数阈值：{TRUE_DEFECT_SCORE_THRESHOLD}")
    
    # 全局归一化+两步筛选：1.分位数初筛 2.过滤误判
    global_diff = global_max - global_min
    if global_diff < 1e-6:
        anomaly_map_norm = np.zeros_like(anomaly_map)
        anomaly_map_thresholded = np.zeros_like(anomaly_map)
    else:
        anomaly_map_norm = (anomaly_map - global_min) / global_diff
        # 第一步：98%分位数初筛
        anomaly_map_thresholded = np.where(anomaly_map_norm >= defect_threshold, anomaly_map_norm, 0)
        # 第二步：过滤浅色误判，仅保留≥真缺陷分数阈值的红色区域
        anomaly_map_thresholded = np.where(
            anomaly_map_thresholded >= TRUE_DEFECT_SCORE_THRESHOLD,
            anomaly_map_thresholded,
            0
        )
    
    return anomaly_map_norm, anomaly_map_thresholded

# -------------------------- 全局热力图生成（原始分辨率，仅保存指定5张大图） --------------------------
def stitch_global_heatmap(
    all_anomaly_maps_norm, all_anomaly_maps_thresholded, all_image_paths, 
    coords_file, original_img_path, save_dir, defect_percentile
):
    """
    生成原始分辨率（14999×3200）的5张大图，完全抛弃matplotlib，直接保存数组
    """
    # 读取坐标
    if not os.path.exists(coords_file):
        raise FileNotFoundError(f"坐标文件不存在：{coords_file}")
    with open(coords_file, "r") as f:
        patch_coords = json.load(f)
    filename_to_coord = {}
    for block_id, items in patch_coords.items():
        for item in items:
            filename_to_coord[item["file"]] = item["coord"]
    
    # 初始化全局矩阵（原始分辨率：14999×3200）
    GLOBAL_HEIGHT = 14999
    GLOBAL_WIDTH = 3200
    global_heatmap_norm = np.zeros((GLOBAL_HEIGHT, GLOBAL_WIDTH), dtype=np.float32)
    global_heatmap_thresholded = np.zeros((GLOBAL_HEIGHT, GLOBAL_WIDTH), dtype=np.float32)
    count_map = np.zeros((GLOBAL_HEIGHT, GLOBAL_WIDTH), dtype=np.float32)
    
    # 填充全局矩阵
    print(f"\n【拼接全局热力图（原始分辨率：{GLOBAL_HEIGHT}×{GLOBAL_WIDTH}）】")
    for img_path, norm_map, thresholded_map in zip(all_image_paths, all_anomaly_maps_norm, all_anomaly_maps_thresholded):
        img_name = os.path.basename(img_path)
        if img_name not in filename_to_coord:
            print(f"⚠️ 未找到{img_name}的坐标，跳过")
            continue
        y1, y2, x1, x2 = filename_to_coord[img_name]
        if norm_map.shape != (512, 512):
            norm_map = np.resize(norm_map, (512, 512))
            thresholded_map = np.resize(thresholded_map, (512, 512))
        global_heatmap_norm[y1:y2, x1:x2] += norm_map
        global_heatmap_thresholded[y1:y2, x1:x2] += thresholded_map
        count_map[y1:y2, x1:x2] += 1.0
    
    # 处理重叠
    count_map[count_map == 0] = 1.0
    global_heatmap_norm = global_heatmap_norm / count_map
    global_heatmap_thresholded = global_heatmap_thresholded / count_map
    
    # 最终过滤：仅保留真缺陷
    global_heatmap_thresholded = np.where(
        global_heatmap_thresholded >= TRUE_DEFECT_SCORE_THRESHOLD,
        global_heatmap_thresholded,
        0
    )
    
    # -------------------------- 1. 读取原始大图（原始分辨率） --------------------------
    if os.path.exists(original_img_path):
        original_img = Image.open(original_img_path).convert("RGB")
        original_img_array = np.array(original_img)
        # 验证分辨率（必须匹配14999×3200）
        if original_img_array.shape[:2] != (GLOBAL_HEIGHT, GLOBAL_WIDTH):
            raise ValueError(f"原始大图分辨率错误！要求{GLOBAL_HEIGHT}×{GLOBAL_WIDTH}，实际{original_img_array.shape[:2]}")
    else:
        raise FileNotFoundError(f"原始缺陷大图不存在：{original_img_path}，无法生成原始分辨率图像")
    
    # -------------------------- 2. 生成5张大图（原始分辨率，无matplotlib） --------------------------
    # 2.1 子图1：原始大图（global_subplot_1_original.png）
    save1 = os.path.join(save_dir, "global_subplot_1_original.png")
    Image.fromarray(original_img_array).save(save1, quality=100, subsampling=0)
    print(f"✅ 原始分辨率子图1保存：{save1}（{GLOBAL_HEIGHT}×{GLOBAL_WIDTH}）")
    
    # 2.2 子图2：全局归一化热力图（global_subplot_2_raw_norm_heatmap.png）
    # 灰度数组→Jet彩色数组（原始分辨率）
    norm_jet_array = apply_jet_colormap(global_heatmap_norm)
    save2 = os.path.join(save_dir, "global_subplot_2_raw_norm_heatmap.png")
    Image.fromarray(norm_jet_array).save(save2, quality=100, subsampling=0)
    print(f"✅ 原始分辨率子图2保存：{save2}（{GLOBAL_HEIGHT}×{GLOBAL_WIDTH}）")
    
    # 2.3 子图3：全局真缺陷热力图（global_subplot_3_true_defect_heatmap.png）
    defect_jet_array = apply_jet_colormap(global_heatmap_thresholded)
    save3 = os.path.join(save_dir, "global_subplot_3_true_defect_heatmap.png")
    Image.fromarray(defect_jet_array).save(save3, quality=100, subsampling=0)
    print(f"✅ 原始分辨率子图3保存：{save3}（{GLOBAL_HEIGHT}×{GLOBAL_WIDTH}）")
    
    # 2.4 子图4：全局真缺陷叠加图（global_subplot_4_true_defect_overlay.png）
    overlay_array = overlay_heatmap_on_image(original_img_array, defect_jet_array, alpha=0.6)
    save4 = os.path.join(save_dir, "global_subplot_4_true_defect_overlay.png")
    Image.fromarray(overlay_array).save(save4, quality=100, subsampling=0)
    print(f"✅ 原始分辨率子图4保存：{save4}（{GLOBAL_HEIGHT}×{GLOBAL_WIDTH}）")
    
    # 2.5 汇总图：4合1（global_heatmap_optimized_summary.png，原始分辨率拼接）
    # 调整子图尺寸为等比例（14999×3200 → 按比例缩小为2×2拼接）
    # 子图尺寸：高度=14999//2，宽度=3200//2（保证比例）
    sub_h = GLOBAL_HEIGHT // 2
    sub_w = GLOBAL_WIDTH // 2
    
    # 缩放4个子图（LANCZOS高质量插值）
    sub1 = Image.fromarray(original_img_array).resize((sub_w, sub_h), Image.Resampling.LANCZOS)
    sub2 = Image.fromarray(norm_jet_array).resize((sub_w, sub_h), Image.Resampling.LANCZOS)
    sub3 = Image.fromarray(defect_jet_array).resize((sub_w, sub_h), Image.Resampling.LANCZOS)
    sub4 = Image.fromarray(overlay_array).resize((sub_w, sub_h), Image.Resampling.LANCZOS)
    
    # 创建汇总画布（2×2）
    summary_w = sub_w * 2
    summary_h = sub_h * 2
    summary_img = Image.new("RGB", (summary_w, summary_h))
    
    # 拼接子图
    summary_img.paste(sub1, (0, 0))
    summary_img.paste(sub2, (sub_w, 0))
    summary_img.paste(sub3, (0, sub_h))
    summary_img.paste(sub4, (sub_w, sub_h))
    
    # 保存汇总图（高质量）
    save5 = os.path.join(save_dir, "global_heatmap_optimized_summary.png")
    summary_img.save(save5, quality=100, subsampling=0)
    print(f"✅ 原始分辨率汇总图保存：{save5}（{summary_h}×{summary_w}）")
    
    return global_heatmap_norm, global_heatmap_thresholded