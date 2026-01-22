import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import gc  # 用于动态卸载模型，节省显存

# 导入自定义工具函数（heatmap_utils.py保持不变）
from heatmap_utils import (
    fix_dinov2_weight_path, generate_valid_heatmap,
    stitch_global_heatmap
)

# -------------------------- 核心配置（关键修改：适配5个Block） --------------------------
# 所有Block的缺陷数据上级目录（包含tire_block1~tire_block5）
TEST_ROOT = "/root/autodl-tmp/Tire/PatchCore_Defect_Patches_512x512"
# 5个模型的保存根目录（包含model_block1~model_block5）
MODEL_SAVE_ROOT = Path("/root/autodl-tmp/Tire/AnomalyDINO/trained_models")
OUTPUT_DIR = Path("/root/autodl-tmp/Tire/AnomalyDINO/final_result")
COORDS_FILE_PATH = "/root/autodl-tmp/Tire/PatchCore_Defect_Patches_512x512/patch_coords.json"
ORIGINAL_DEFECT_IMG_PATH = "/root/autodl-tmp/Tire/Data/PreProcess/Original/defect.png"

# 缺陷筛选配置（与训练时一致）
DEFECT_PERCENTILE = 99
TRUE_DEFECT_SCORE_THRESHOLD = 0.7  # 可调整的真缺陷阈值
BLOCK_IDS = [1, 2, 3, 4, 5]  # 5个Block的ID

# 设备配置
if torch.cuda.is_available():
    GPU_INDEX = 0
    DEVICE = torch.device(f"cuda:{GPU_INDEX}")
    torch.cuda.set_device(GPU_INDEX)
    print(f"【设备信息】使用GPU：cuda:{GPU_INDEX} | 名称：{torch.cuda.get_device_name(GPU_INDEX)}")
else:
    DEVICE = torch.device("cpu")
    print("【设备信息】使用CPU（无可用GPU）")

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------- 初始化配置 --------------------------
fix_dinov2_weight_path()
torch.set_float32_matmul_precision('medium')

# -------------------------- 核心工具：加载单个Block的模型 --------------------------
def load_block_model(block_id):
    """加载指定Block的模型（从训练好的.pth文件）"""
    model_path = MODEL_SAVE_ROOT / f"model_block{block_id}" / f"anomalydino_tire_model_block{block_id}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Block {block_id} 的模型文件不存在：{model_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    image_size = checkpoint["image_size"]
    pre_processor = checkpoint["pre_processor"]
    
    # 初始化模型
    from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
    model = AnomalyDINO(
        num_neighbours=1,  # 与训练时一致
        encoder_name="dinov2_vit_small_14",  # 与训练时一致
        masking=False,
        coreset_subsampling=False,
        sampling_ratio=0.1,  # 与训练时一致
        pre_processor=pre_processor,
        post_processor=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.model.memory_bank = checkpoint["memory_bank"].to(DEVICE)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ 加载Block {block_id} 模型成功！特征库尺寸：{model.model.memory_bank.shape}")
    return model, pre_processor, image_size

# -------------------------- 步骤1：收集所有Block的缺陷patch（带BlockID标记） --------------------------
def collect_defect_patches_with_block():
    """遍历5个Block的test/defect目录，收集 (patch路径, block_id)"""
    defect_patches = []
    for block_id in BLOCK_IDS:
        # 每个Block的缺陷目录路径
        block_defect_dir = f"{TEST_ROOT}/tire_block{block_id}/test/defect"
        if not os.path.exists(block_defect_dir):
            print(f"⚠️ Block {block_id} 的缺陷目录不存在：{block_defect_dir}，跳过该Block")
            continue
        
        # 收集该Block下的所有缺陷patch
        block_patches = [
            (os.path.join(block_defect_dir, f), block_id)
            for f in os.listdir(block_defect_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        defect_patches.extend(block_patches)
        print(f"✅ Block {block_id} 收集到 {len(block_patches)} 个缺陷patch")
    
    if not defect_patches:
        raise ValueError("所有Block均未找到缺陷patch！")
    print(f"\n【推理配置】总计收集到 {len(defect_patches)} 个缺陷patch")
    return defect_patches

# 执行收集
defect_patches = collect_defect_patches_with_block()  # 格式：[(path1, block1), (path2, block2), ...]
all_image_paths = [p[0] for p in defect_patches]  # 所有patch路径
all_block_ids = [p[1] for p in defect_patches]    # 每个patch对应的BlockID

# -------------------------- 步骤2：批量推理（按Block分流，动态加载模型） --------------------------
all_anomaly_maps = []  # 存储所有patch的异常图（不管哪个Block）
current_block_id = None
current_model = None
current_pre_processor = None
current_image_size = None

print("\n【阶段1：批量推理（按Block分流）】")
for idx, (image_path, block_id) in enumerate(defect_patches):
    # 切换模型：当前Block与上一个不同时，加载新模型（卸载旧模型节省显存）
    if block_id != current_block_id:
        # 卸载旧模型（关键：释放显存）
        if current_model is not None:
            del current_model, current_pre_processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"🔧 已卸载Block {current_block_id} 模型，释放显存")
        
        # 加载当前Block的模型
        current_model, current_pre_processor, current_image_size = load_block_model(block_id)
        current_block_id = block_id
    
    # 图像预处理（使用当前Block模型的专属pre_processor）
    image = Image.open(image_path).convert("RGB")
    image = current_pre_processor(image).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    
    # 推理（禁用梯度计算，节省显存）
    with torch.no_grad():
        inference_result = current_model(image)
        anomaly_map = inference_result.anomaly_map
    
    # 处理异常图（统一调整为训练时的图像尺寸）
    anomaly_map_np = anomaly_map.cpu().detach().numpy()
    anomaly_map_np = np.squeeze(anomaly_map_np)
    if anomaly_map_np.shape != current_image_size:
        anomaly_map_np = np.resize(anomaly_map_np, current_image_size)
    
    # 收集结果
    all_anomaly_maps.append(anomaly_map_np)
    print(f"  已处理样本 {idx+1}/{len(defect_patches)}：{os.path.basename(image_path)}（Block {block_id}）")

# 推理完成后，彻底卸载所有模型
del current_model, current_pre_processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -------------------------- 步骤3：计算全局极值和阈值（统一所有Block的结果） --------------------------
all_anomaly_flat = np.concatenate([am.flatten() for am in all_anomaly_maps])
global_min = all_anomaly_flat.min()
global_max = all_anomaly_flat.max()
defect_threshold = np.percentile(all_anomaly_flat, DEFECT_PERCENTILE)

print(f"\n【全局极值与阈值统计】")
print(f"  全局最小值：{global_min:.6f} | 全局最大值：{global_max:.6f}")
print(f"  {DEFECT_PERCENTILE}%分位数阈值：{defect_threshold:.6f}")
print(f"  真缺陷分数阈值：{TRUE_DEFECT_SCORE_THRESHOLD}")

# -------------------------- 步骤4：生成单个小块热力图（保持原始逻辑） --------------------------
all_anomaly_maps_norm = []
all_anomaly_maps_thresholded = []
print("\n【阶段2：生成单个小块热力图】")

# 所有Block的图像尺寸应该一致（训练时都是512x512），取第一个即可
IMAGE_SIZE = current_image_size  # 或直接写(512,512)，确保与训练一致

for image_path, anomaly_map in zip(all_image_paths, all_anomaly_maps):
    img_name = os.path.basename(image_path).replace(".png", "_heatmap.png").replace(".jpg", "_heatmap.png")
    save_path = OUTPUT_DIR / img_name
    
    # 生成热力图（调用工具函数，不单独保存子图）
    norm_map, thresholded_map = generate_valid_heatmap(
        image_path=image_path,
        anomaly_map=torch.from_numpy(anomaly_map),
        save_path=save_path,
        global_min=global_min,
        global_max=global_max,
        defect_threshold=defect_threshold,
        image_size=IMAGE_SIZE,
        defect_percentile=DEFECT_PERCENTILE
    )
    
    # 存储结果（用于后续全局拼接）
    all_anomaly_maps_norm.append(norm_map)
    all_anomaly_maps_thresholded.append(thresholded_map)
    print(f"  已生成热力图：{os.path.basename(save_path)}")

# -------------------------- 步骤5：生成原始分辨率全局热力图（保持原始逻辑） --------------------------
try:
    stitch_global_heatmap(
        all_anomaly_maps_norm=all_anomaly_maps_norm,
        all_anomaly_maps_thresholded=all_anomaly_maps_thresholded,
        all_image_paths=all_image_paths,
        coords_file=COORDS_FILE_PATH,
        original_img_path=ORIGINAL_DEFECT_IMG_PATH,
        save_dir=OUTPUT_DIR,
        defect_percentile=DEFECT_PERCENTILE
    )
except Exception as e:
    print(f"⚠️ 生成全局热力图时出错：{e}")
    import traceback
    traceback.print_exc()

# -------------------------- 完成提示 --------------------------
print(f"\n🎉 所有任务完成！")
print(f"   📂 输出目录：{OUTPUT_DIR}")
print(f"\n📌 生成的原始分辨率大图清单：")
print(f"   1. global_subplot_1_original.png          （原始缺陷大图，14999×3200）")
print(f"   2. global_subplot_2_raw_norm_heatmap.png  （全局归一化热力图，14999×3200）")
print(f"   3. global_subplot_3_true_defect_heatmap.png（全局真缺陷热力图，14999×3200）")
print(f"   4. global_subplot_4_true_defect_overlay.png（全局真缺陷叠加图，14999×3200）")
print(f"   5. global_heatmap_optimized_summary.png   （4合1汇总图，高质量拼接）")
print(f"\n📌 关键说明：")
print(f"   - 已使用5个Block专属模型分别推理，结果统一拼接；")
print(f"   - 真缺陷阈值可调整：TRUE_DEFECT_SCORE_THRESHOLD = {TRUE_DEFECT_SCORE_THRESHOLD}；")
print(f"   - 误判多则调高阈值，缺陷漏检则调低阈值（建议范围0.5~0.9）。")