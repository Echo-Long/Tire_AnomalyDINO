import cv2
import numpy as np
import os
from pathlib import Path
import json

def split_defect_tire_3200x14999_to_512x512(
    input_defect_img_path: str,
    save_root: str,
    block_num: int = 5,
    patch_size: int = 512,  # 修改为512×512
    stride: int = 256       # 步长256，重叠50%
):
    """
    核心修改：
    1. 裁切512×512小块，重叠50%（步长256）
    2. 边界不足时不填充任意像素，仅保留图像内部完整块
    3. 保持和正样本一致的大块分割、目录结构、坐标记录逻辑
    用途：裁切负样本（缺陷图），适配模型推理
    """
    # 1. 读取缺陷图（兼容灰度/彩色图）
    print(f"🔍 读取缺陷图：{input_defect_img_path}")
    img = cv2.imread(input_defect_img_path)
    if img is None:
        raise ValueError(f"无法读取缺陷图！路径：{input_defect_img_path}")
    
    # 统一转为单通道灰度图（和训练数据格式一致）
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    print(f"✅ 缺陷图维度：高度={h}，宽度={w}（单通道灰度图）")
    assert w == 3200, f"原图宽度必须为3200，当前为{w}"
    assert h == 14999, f"原图高度必须为14999，当前为{h}"

    # 2. 沿高度分割为5个宽3200的大块（保持原图方向）
    block_height = h // block_num  # 14999//5=2999
    remainder = h % block_num      # 14999%5=4，最后一块多4像素
    print(f"📌 沿高度分割为{block_num}个大块：")
    print(f"   大块维度：宽度={w}，前{block_num-1}块高度={block_height}，最后一块高度={block_height+remainder}")

    # 3. 逐个处理大块，裁切512×512小块（记录坐标，用于后续拼接）
    patch_coords_dict = {}  # 保存每个小块的坐标：{block_id: [(y1,y2,x1,x2), ...]}
    for block_idx in range(block_num):
        block_id = block_idx + 1
        patch_coords_dict[block_id] = []
        print(f"\n===== 处理第{block_id}个大块 =====")
        # 大块坐标（高度方向分割）
        start_y = block_idx * block_height
        end_y = start_y + block_height + (remainder if block_idx == block_num-1 else 0)
        block_img = img[start_y:end_y, :]  # 前4块：2999x3200，最后一块：3003x3200
        block_h, block_w = block_img.shape
        print(f"   大块{block_id}维度：{block_h}×{block_w} (h×w)")

        # 创建并准备保存目录
        dataset_dir = Path(save_root) / f"tire_block{block_id}"
        block_save_dir = dataset_dir
        block_save_dir.mkdir(parents=True, exist_ok=True)
        # 保存大块原图
        block_img_3ch = np.repeat(block_img[:, :, np.newaxis], 3, axis=2)
        block_img_path = block_save_dir / f"block{block_id}_full.png"
        cv2.imwrite(str(block_img_path), block_img_3ch)
        print(f"   已保存大块图：{block_img_path}")
        
        # 4. 创建缺陷图保存目录（MVTec格式：test/defect）
        test_defect_dir = dataset_dir / "test" / "defect"
        test_defect_dir.mkdir(parents=True, exist_ok=True)
        print(f"   小块保存路径：{test_defect_dir}")

        # 5. 裁切512×512小块（核心：仅保留内部完整块，无填充）
        patch_count = 0
        # 循环终止条件：确保小块右下角不超出大块边界（无填充）
        for y in range(0, block_h - patch_size + 1, stride):
            for x in range(0, block_w - patch_size + 1, stride):
                # 记录小块在全图中的全局坐标
                y1_local, y2_local = y, y + patch_size
                x1_local, x2_local = x, x + patch_size
                y1, y2 = start_y + y1_local, start_y + y2_local
                x1, x2 = x1_local, x2_local

                # 提取小块（仅内部完整像素，无填充）
                patch = block_img[y1_local:y2_local, x1_local:x2_local]
                # 验证小块尺寸（确保无填充）
                assert patch.shape == (patch_size, patch_size), f"小块尺寸异常：{patch.shape}，应为({patch_size},{patch_size})"

                # 单通道转3通道（适配模型输入）
                patch_3ch = np.repeat(patch[:, :, np.newaxis], 3, axis=2)

                # 保存512×512×3小块（命名和正样本一致）
                patch_name = f"block{block_id}_patch{patch_count}.png"
                patch_path = test_defect_dir / patch_name
                cv2.imwrite(str(patch_path), patch_3ch)
                
                # 记录映射信息（全局坐标）
                patch_coords_dict[block_id].append((patch_name, [y1, y2, x1, x2]))
                patch_count += 1

        print(f"✅ 大块{block_id}分割完成：生成{patch_count}个512×512×3小块（无填充）")

    # 6. 保存坐标文件（用于后续热力图拼接）
    coords_save_path = Path(save_root) / "patch_coords.json"
    # 将坐标转为可序列化格式
    coords_serializable = {
        k: [{"file": item[0], "coord": [int(i) for i in item[1]]} for item in v]
        for k, v in patch_coords_dict.items()
    }
    with open(coords_save_path, "w") as f:
        json.dump(coords_serializable, f, indent=2)

    # 保存每个block的坐标映射文件
    for block_id, items in patch_coords_dict.items():
        block_map_path = Path(save_root) / f"tire_block{block_id}" / "patch_map.json"
        block_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(block_map_path, "w") as bf:
            json.dump([{"file": it[0], "coord": it[1]} for it in items], bf, indent=2)
    
    # 7. 最终验证
    print(f"\n🎉 缺陷图分割完成！")
    print(f"   📂 小块保存根目录：{save_root}")
    print(f"   📄 坐标文件路径：{coords_save_path}")
    first_patch = Path(save_root) / "tire_block1/test/defect/block1_patch0.png"
    if first_patch.exists():
        check_img = cv2.imread(str(first_patch))
        print(f"✅ 验证：第一个小块维度={check_img.shape}（应为(512,512,3)）")

if __name__ == "__main__":
    # 配置路径（修改为你的实际路径）
    INPUT_DEFECT_IMG_PATH = "/root/autodl-tmp/Tire/Data/PreProcess/Original/defect.png"  # 缺陷图路径
    SAVE_ROOT = "/root/autodl-tmp/Tire/PatchCore_Defect_Patches_512x512"  # 小块保存根目录
    
    # 执行缺陷图分割（512×512，重叠50%，无填充）
    split_defect_tire_3200x14999_to_512x512(
        input_defect_img_path=INPUT_DEFECT_IMG_PATH,
        save_root=SAVE_ROOT,
        block_num=5,
        patch_size=512,
        stride=256
    )