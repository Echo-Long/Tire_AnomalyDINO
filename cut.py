import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, List
import json

def split_tire_3200x14999_to_512x512_augmented_20px(
    input_img_path: str,
    save_root: str,
    block_num: int = 5,
    patch_size: int = 512,  # 裁切512×512小块
    stride: int = 256,      # 步长256，重叠50%
    shift_pixels: int = 20  # 固定平移20像素（贴合工业微小偏移）
):
    """
    工业场景定制版：
    1. 每个原始512×512小块，生成上/下/左/右各20像素偏移的4个增强图
    2. 增强图与原图保存在同一目录，文件名标识平移方向
    3. 保留完整坐标记录（含平移修正），无填充，仅保留完整块
    """
    # 1. 读取原始灰度图
    print(f"🔍 读取原始灰度图：{input_img_path}")
    img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图片！路径：{input_img_path}")
    h, w = img.shape
    print(f"✅ 原始图维度：高度={h}，宽度={w}（单通道灰度图）")
    assert w == 3200, f"原图宽度必须为3200，当前为{w}"
    assert h == 14999, f"原图高度必须为14999，当前为{h}"
    
    # 验证平移参数（20像素远小于步长和patch_size，保证完整性）
    assert 0 < shift_pixels < min(stride, patch_size), \
        f"平移像素数({shift_pixels})需小于步长({stride})和小块尺寸({patch_size})"

    # 2. 沿高度分割为5个宽3200的大块
    block_height = h // block_num  # 14999//5=2999
    remainder = h % block_num      # 14999%5=4，最后一块多4像素
    print(f"📌 沿高度分割为{block_num}个大块，每个小块生成4个20像素平移增强图（上/下/左/右）")
    print(f"   大块维度：宽度={w}，前{block_num-1}块高度={block_height}，最后一块高度={block_height+remainder}")

    # 3. 逐个处理大块，生成带20像素平移的增强小块
    patch_coords_dict = {}
    for block_idx in range(block_num):
        block_id = block_idx + 1
        patch_coords_dict[block_id] = []
        print(f"\n===== 处理第{block_id}个大块 =====")

        # 大块坐标（高度方向）
        start_y = block_idx * block_height
        end_y = start_y + block_height + (remainder if block_idx == block_num-1 else 0)
        block_img = img[start_y:end_y, :]  # 前4块：2999x3200，最后一块：3003x3200
        block_h, block_w = block_img.shape
        print(f"   大块{block_id}原始维度：{block_h}×{block_w} (h×w)")

        # 保存大块原图（便于对比）
        dataset_dir = Path(save_root) / f"tire_block{block_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        block_img_3ch = np.repeat(block_img[:, :, np.newaxis], 3, axis=2)
        block_img_path = dataset_dir / f"block{block_id}_full.png"
        cv2.imwrite(str(block_img_path), block_img_3ch)
        print(f"   已保存大块原图：{block_img_path}")

        # 4. 创建保存目录（增强图与原图同目录）
        train_good_dir = dataset_dir / "train" / "good"
        test_defect_dir = dataset_dir / "test" / "defect"
        train_good_dir.mkdir(parents=True, exist_ok=True)
        test_defect_dir.mkdir(parents=True, exist_ok=True)
        print(f"   小块保存路径：{test_defect_dir} (test) 和 {train_good_dir} (train)")

        # 5. 裁切原始512×512小块，并为每个小块生成4个20像素平移增强图
        total_patch_count = 0  # 统计所有小块（原图+增强图）
        # 循环终止条件：确保原始小块右下角不超出大块边界
        for y in range(0, block_h - patch_size + 1, stride):
            for x in range(0, block_w - patch_size + 1, stride):
                # 原始小块的局部坐标
                y1_local, y2_local = y, y + patch_size
                x1_local, x2_local = x, x + patch_size
                # 提取原始512×512小块
                original_patch = block_img[y1_local:y2_local, x1_local:x2_local]
                assert original_patch.shape == (patch_size, patch_size), f"原始小块尺寸异常：{original_patch.shape}"
                original_patch_3ch = np.repeat(original_patch[:, :, np.newaxis], 3, axis=2)

                # 原始小块文件名
                patch_base_name = f"block{block_id}_patch{total_patch_count // 5}"  # 每5个（1原+4增强）共用一个基础编号
                # 保存原始小块
                original_patch_name = f"{patch_base_name}_original.png"
                cv2.imwrite(str(train_good_dir / original_patch_name), original_patch_3ch)
                cv2.imwrite(str(test_defect_dir / original_patch_name), original_patch_3ch)

                # 记录原始小块坐标（全局）
                patch_coords_dict[block_id].append({
                    "file": original_patch_name,
                    "shift_dir": "original",
                    "shift_pixels": 0,
                    "coord": [int(start_y + y1_local), int(start_y + y2_local), int(x1_local), int(x2_local)]
                })
                total_patch_count += 1

                # ========== 核心：为当前小块生成4个20像素平移增强图 ==========
                shift_configs = [
                    ("up", shift_pixels, 0),       # 上移20像素：y方向偏移+20
                    ("down", -shift_pixels, 0),     # 下移20像素：y方向偏移-20
                    ("left", 0, shift_pixels),      # 左移20像素：x方向偏移+20
                    ("right", 0, -shift_pixels)     # 右移20像素：x方向偏移-20
                ]

                for shift_dir, y_shift, x_shift in shift_configs:
                    # 计算增强小块的原始大块内坐标（需确保仍在大块范围内）
                    aug_y1 = y1_local + y_shift
                    aug_y2 = y2_local + y_shift
                    aug_x1 = x1_local + x_shift
                    aug_x2 = x2_local + x_shift

                    # 校验增强小块是否完全在大块内（避免越界）
                    if (aug_y1 >= 0 and aug_y2 <= block_h) and (aug_x1 >= 0 and aug_x2 <= block_w):
                        # 提取平移后的增强小块
                        aug_patch = block_img[aug_y1:aug_y2, aug_x1:aug_x2]
                        aug_patch_3ch = np.repeat(aug_patch[:, :, np.newaxis], 3, axis=2)

                        # 增强小块文件名（标识平移方向）
                        aug_patch_name = f"{patch_base_name}_shift{shift_dir}{shift_pixels}.png"
                        # 保存增强小块（与原图同目录）
                        cv2.imwrite(str(train_good_dir / aug_patch_name), aug_patch_3ch)
                        cv2.imwrite(str(test_defect_dir / aug_patch_name), aug_patch_3ch)

                        # 记录增强小块的全局坐标（修正平移偏移）
                        patch_coords_dict[block_id].append({
                            "file": aug_patch_name,
                            "shift_dir": shift_dir,
                            "shift_pixels": shift_pixels,
                            "coord": [
                                int(start_y + aug_y1),
                                int(start_y + aug_y2),
                                int(aug_x1),
                                int(aug_x2)
                            ]
                        })
                        total_patch_count += 1
                    else:
                        # 边缘小块平移后越界则跳过（保证所有保存的块都是完整的）
                        print(f"   ⚠️ 小块{patch_base_name}向{shift_dir}平移20像素后越界，跳过")

        print(f"✅ 大块{block_id}分割完成：共生成{total_patch_count}个小块（含原始块+20像素平移增强块）")

    # 6. 保存坐标文件（含平移信息，用于后续拼接热力图）
    coords_save_path = Path(save_root) / "patch_coords_20px_augmented.json"
    with open(coords_save_path, "w") as f:
        json.dump(patch_coords_dict, f, indent=2)

    # 保存每个大块的坐标映射
    for block_id, items in patch_coords_dict.items():
        block_map_path = Path(save_root) / f"tire_block{block_id}" / "patch_map_20px_augmented.json"
        with open(block_map_path, "w") as bf:
            json.dump(items, bf, indent=2)

    print(f"\n🎉 所有增强分割完成！数据集路径：{save_root}")
    # 验证第一个原始小块尺寸
    first_original_patch = Path(save_root) / "tire_block1/train/good/block1_patch0_original.png"
    if first_original_patch.exists():
        check_img = cv2.imread(str(first_original_patch))
        print(f"✅ 验证：第一个原始小块维度={check_img.shape}（应为(512,512,3)）")
    # 验证第一个增强小块尺寸
    first_aug_patch = Path(save_root) / "tire_block1/train/good/block1_patch0_shiftup20.png"
    if first_aug_patch.exists():
        check_img = cv2.imread(str(first_aug_patch))
        print(f"✅ 验证：第一个增强小块维度={check_img.shape}（应为(512,512,3)）")

if __name__ == "__main__":
    # 配置路径（修改为你的实际路径）
    INPUT_IMG_PATH = "/root/autodl-tmp/Tire/Data/PreProcess/Original/241163281418_2_2_20251230_172410_luminancecrop_gray.png"
    SAVE_ROOT = "/root/autodl-tmp/Tire/PatchCore_Dataset_512x512_20px_augmented"
    
    # 执行20像素平移增强裁切（无需修改其他参数）
    split_tire_3200x14999_to_512x512_augmented_20px(
        input_img_path=INPUT_IMG_PATH,
        save_root=SAVE_ROOT,
        block_num=5,
        patch_size=512,
        stride=256,
        shift_pixels=20  # 固定20像素平移
    )