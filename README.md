# Readme

## 前言

基于`anomalib`​中的`anomalyDINO`模型的工业缺陷检测。

## 使用方法

### 环境搭建

创建云服务器，环境为：(实测`pytorch1.x和python3.10`及以下均不能运行）

```python
PyTorch  2.1.2

Python  3.10(ubuntu22.04)

CUDA  11.8
```

conda创建虚拟环境，依照[指示](https://github.com/open-edge-platform/anomalib)来安装前置包。

```python
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib

# Install in editable mode with a specific backend
pip install -e ".[cpu]"
```

此时应该安装的`anomalib v2.3.0.dev0`​，`lightning v2.0.7`

### 准备数据集

运行`cut.py`​和`cut_defect.py`​进行正样本和待检测样本的切块，可自行微调`cut.py`​和`cut_defect.py`的路径，切块个数 $n$ ，切块大小等，构建符合要求的数据集。

### 训练

运行`download.py`​下载`dinov2_vits14_pretrain.pth`​文件，自动放到指定的文件夹里（`/root/autodl-tmp/.cache/torch/hub/checkpoints/`​），也可以修改`train.py`进行自定义路径。

调整`train.py`中的路径，切块数 $n$ 等，运行`train.py`进行训练，自动在对应路径生成 $n$ 个模型。

调整`predict.py`中的路径，切块数 $n$，筛选阈值等，运行`predict.py`进行预测，自动生成原图、原始热力图、阈值筛选热力图、蒙版热力图、四合一图。

如果也想生成单个小块的中间热力图，可以去`generate_all`​文件夹中替换`predict.py`​和`heatmap_utils.py`的生成逻辑

> [!IMPORTANT]
> ​`generate_all`​仅能训练单`block`​，没有为多`block`作出适配

‍

‍

‍

‍

‍

‍

‍

‍
