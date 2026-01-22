import os
import torch
import numpy as np
import gc  # 导入垃圾回收模块
from pathlib import Path
from PIL import Image

# -------------------------- 关键：全局指定缓存/临时文件到数据盘 --------------------------
os.environ['TORCH_HOME'] = "/root/autodl-tmp/.cache/torch"
os.environ['LIGHTNING_LOGS'] = "/root/autodl-tmp/lightning_logs"
os.environ['TMPDIR'] = "/root/autodl-tmp/tmp"
# 创建临时目录
Path("/root/autodl-tmp/tmp").mkdir(exist_ok=True, parents=True)
Path("/root/autodl-tmp/.cache/torch").mkdir(exist_ok=True, parents=True)

# -------------------------- 基础配置 --------------------------
from anomalib.models.components.dinov2.dinov2_loader import DinoV2Loader
def fixed_get_weight_path(self, model_type, architecture, patch_size):
    return Path("/root/autodl-tmp/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
DinoV2Loader._get_weight_path = fixed_get_weight_path
print("✅ 已强制覆盖DinoV2Loader的权重路径计算逻辑")

torch.set_float32_matmul_precision('medium')

# -------------------------- 核心路径配置 --------------------------
TRAIN_ROOT = "/root/autodl-tmp/Tire/PatchCore_Dataset_512x512_20px_augmented"
MODEL_SAVE_ROOT = Path("/root/autodl-tmp/Tire/AnomalyDINO/trained_models")
IMAGE_SIZE = (512, 512)
BLOCK_IDS = [1,2,3,4,5]

# -------------------------- 设备配置 --------------------------
if torch.cuda.is_available():
    GPU_INDEX = 0
    DEVICE = torch.device(f"cuda:{GPU_INDEX}")
    torch.cuda.set_device(GPU_INDEX)
    print(f"【设备信息】使用GPU：cuda:{GPU_INDEX} | 名称：{torch.cuda.get_device_name(GPU_INDEX)}")
else:
    DEVICE = torch.device("cpu")
    print("【设备信息】使用CPU（无可用GPU）")

MODEL_SAVE_ROOT.mkdir(exist_ok=True, parents=True)

# -------------------------- 数据预处理配置 --------------------------
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize

custom_transform = Compose([
    Resize(size=IMAGE_SIZE),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
custom_pre_processor = PreProcessor(transform=custom_transform)

# -------------------------- 循环训练5个block的模型 --------------------------
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import shutil

# 清理旧的Lightning日志
lightning_logs_path = Path("/root/autodl-tmp/lightning_logs")
if lightning_logs_path.exists():
    shutil.rmtree(lightning_logs_path)
print("✅ 已清理旧的Lightning日志")

for block_id in BLOCK_IDS:
    print(f"\n" + "="*50)
    print(f"开始训练 Block {block_id} 的模型")
    print("="*50)
    
    # 1. 动态路径（每个block完全独立）
    TRAIN_NORMAL_ROOT = f"{TRAIN_ROOT}/tire_block{block_id}/train"
    MODEL_SAVE_DIR = MODEL_SAVE_ROOT / f"model_block{block_id}"
    TMP_CKPT_DIR = MODEL_SAVE_DIR / "tmp_ckpt"
    MODEL_SAVE_DIR.mkdir(exist_ok=True, parents=True)
    
    # 2. 样本校验
    good_dir = f"{TRAIN_NORMAL_ROOT}/good"
    if not os.path.exists(good_dir):
        raise ValueError(f"Block {block_id} 的正常样本目录不存在：{good_dir}")
    normal_sample_count = len([f for f in os.listdir(good_dir) if f.endswith(('.png','.jpg','.jpeg'))])
    print(f"【样本校验】Block {block_id} 正常样本数：{normal_sample_count}")
    assert normal_sample_count > 0, f"Block {block_id} 正常样本数不能为0！"
    
    # 3. 构建DataModule
    datamodule = Folder(
        name=f"tire_anomaly_block{block_id}",
        root=TRAIN_NORMAL_ROOT,
        normal_dir="good",
        abnormal_dir="",
        normal_split_ratio=0.0,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=0.0,
        train_batch_size=8,
        eval_batch_size=1,
        num_workers=1,
        augmentations=custom_pre_processor,
        extensions=(".png", ".jpg", ".jpeg"),
    )
    datamodule.setup()
    
    # 4. 初始化模型
    model = AnomalyDINO(
        num_neighbours=1,
        encoder_name="dinov2_vit_small_14",
        masking=False,
        coreset_subsampling=False,
        sampling_ratio=0.1,
        pre_processor=custom_pre_processor,
        post_processor=True,
    )
    model.save_hyperparameters(ignore=['pre_processor'])
    model = model.to(DEVICE)
    print(f"✅ Block {block_id} 模型已迁移到目标设备")
    
    # 5. 配置ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=TMP_CKPT_DIR,
        filename=f"block{block_id}",
        save_top_k=0,
        save_last=False,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
    
    # 6. 初始化Trainer
    trainer = Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[GPU_INDEX] if torch.cuda.is_available() else 1,
        gradient_clip_val=0,
        max_epochs=1,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        default_root_dir="/root/autodl-tmp/lightning_logs",
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )
    
    # 7. 训练
    print(f"\n【Block {block_id}】开始构建正常样本特征库")
    trainer.fit(model=model, datamodule=datamodule)
    
    # 8. 保存模型（关键：保存到硬盘，后续可加载）
    try:
        memory_bank = model.model.memory_bank.to(DEVICE, non_blocking=True)
        print(f"✅ Block {block_id} 特征库构建成功！尺寸：{memory_bank.shape} | 设备：{memory_bank.device}")
        assert memory_bank.shape[0] > 0, f"Block {block_id} 特征库为空！"
        
        # 保存到硬盘（这一步是“保留模型”的核心，文件不会被删除）
        model_save_path = MODEL_SAVE_DIR / f"anomalydino_tire_model_block{block_id}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "memory_bank": memory_bank,
            "pre_processor": custom_pre_processor,
            "image_size": IMAGE_SIZE,
            "block_id": block_id
        }, model_save_path)
        print(f"✅ Block {block_id} 模型已保存到：{model_save_path}")
        
        # 清理临时目录
        if TMP_CKPT_DIR.exists():
            shutil.rmtree(TMP_CKPT_DIR)
            
    except Exception as e:
        raise RuntimeError(f"❌ Block {block_id} 特征库构建/保存失败：{e}")
    
    # -------------------------- 核心修改：清理显存（不删除硬盘模型文件） --------------------------
    if torch.cuda.is_available():
        # 1. 删除当前Block的内存实例（模型、数据、训练器）—— 释放Python引用
        del model, datamodule, trainer, memory_bank
        # 2. 强制垃圾回收（回收Python层面的内存）
        gc.collect()
        # 3. 清空GPU缓存（释放显存）
        torch.cuda.empty_cache()
        # 验证显存释放情况（可选，用于调试）
        free_mem = torch.cuda.get_device_properties(DEVICE).total_memory - torch.cuda.memory_allocated(DEVICE)
        print(f"✅ Block {block_id} 显存已清理 | 释放后空闲显存：{free_mem / 1024**3:.2f} GB")
    else:
        del model, datamodule, trainer, memory_bank
        gc.collect()
        print(f"✅ Block {block_id} 内存已清理")

# 清理全局临时文件
if lightning_logs_path.exists():
    shutil.rmtree(lightning_logs_path)

print("\n" + "="*50)
print("🎉 所有5个Block的模型训练完成！")
print(f"模型保存根目录：{MODEL_SAVE_ROOT}")
print("="*50)