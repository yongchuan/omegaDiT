# OmegaDiT: 基于 VA-VAE 的文本条件扩散模型

**仅 130M 参数，即可实现高质量文本到图像生成。** OmegaDiT 是一个轻量级扩散模型，证明了小参数量模型在合理的架构设计和训练策略下，同样能够生成高质量的图像。

## 生成效果展示

<table>
  <tr>
    <td width="25%"><img src="demos/1.png" width="100%"></td>
    <td width="25%"><img src="demos/2.png" width="100%"></td>
    <td width="25%"><img src="demos/3.png" width="100%"></td>
    <td width="25%"><img src="demos/4.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><sub>Happy dreamy owl monster sitting on a tree branch, colorful glittering particles, forest background, detailed feathers.</sub></td>
    <td align="center"><sub>Game-Art - An island with different geographical properties and multiple small cities floating in space</sub></td>
    <td align="center"><sub>A cyberpunk panda is taking a walk on the street</sub></td>
    <td align="center"><sub>Half human, half robot, repaired human</sub></td>
  </tr>
  <tr>
    <td width="25%"><img src="demos/5.png" width="100%"></td>
    <td width="25%"><img src="demos/6.png" width="100%"></td>
    <td width="25%"><img src="demos/7.png" width="100%"></td>
    <td width="25%"><img src="demos/8.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><sub>porcelain girl's face. fine texture. surreal</sub></td>
    <td align="center"><sub>Poster of a mechanical cat, techical Schematics viewed from front.</sub></td>
    <td align="center"><sub>a car runing on the sand.</sub></td>
    <td align="center"><sub>a highly detailed anime sexy beauty.</sub></td>
  </tr>
  <tr>
    <td width="25%"><img src="demos/9.png" width="100%"></td>
    <td width="25%"><img src="demos/10.png" width="100%"></td>
    <td width="25%"><img src="demos/11.png" width="100%"></td>
    <td width="25%"><img src="demos/12.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><sub>A cat</sub></td>
    <td align="center"><sub>fruit cream cake</sub></td>
    <td align="center"><sub>A cat holding a sign that says hello world</sub></td>
    <td align="center"><sub>a dog</sub></td>
  </tr>
</table>

---

本项目是基于 [SpeedrunDiT](https://github.com/SwayStar123/SpeedrunDiT/) 改造的新版本，主要改进包括：引入 VA-VAE (Vision Foundation Model Aligned VAE) 替代原有的 INVAE，新增文本标签条件支持，以及 CLIP 文本编码器集成。

> **Java 开发者？** 本模型已在 [Omega-AI](https://gitee.com/dromara/omega-ai) 框架中完整实现，支持纯 Java 环境下的推理与训练。 [查看详情 >>](#omega-ai-java-深度学习框架实现)

## 主要特性

- **VA-VAE 编码器**: 使用 DINOv2 对齐的视觉基础模型 VAE，提供更好的语义表达能力
- **多模态条件支持**: 同时支持 ImageNet 类标签和自由文本描述作为条件
- **CLIP 文本编码**: 集成 ModelScope CLIP-GmP-ViT-L-14 进行文本特征提取
- **DINOv2 特征对齐**: 通过投影损失对齐 DINOv2 特征，增强语义理解
- **对比流匹配 (CFM)**: 改进的损失函数设计，提升生成质量

## 项目结构

```
OmegaDiT/
├── train.py                    # 核心训练脚本 (Accelerate 分布式)
├── generate.py                 # 多 GPU 采样脚本
├── loss.py                     # 损失函数 (扩散 + 投影 + CFM)
├── dataset.py                  # ImageNet 类标签数据集
├── json_label_dataset.py       # 文本标签数据集 (新增)
├── samplers.py                 # Euler-Maruyama 采样器
├── utils.py                    # 工具函数
│
├── models/                     # 模型架构
│   ├── sit.py                  # SiT 核心架构
│   ├── vavae.py                # VA-VAE 实现 (新增)
│   ├── autoencoder.py          # AutoencoderKL
│   ├── invae.py                # INVAE 实现
│   └── pos_embed.py            # 旋转位置嵌入 (RoPE)
│
├── preprocessing/              # 数据预处理
│   ├── dataset_tools.py        # ImageNet 转换和编码
│   ├── encoders.py             # 编码器加载 (DINOv2/VA-VAE)
│   └── README.md               # 预处理指南
│
├── evaluations/                # 评估工具
│   └── evaluator.py            # FID/sFID/IS/Precision/Recall
│
├── tokenizer/configs/          # 分词器配置
│   └── vavae_f16d32_vfdinov2.yaml
│
├── train.sh                    # 训练脚本示例
├── eval.sh                     # 评估脚本示例
└── requirements.txt            # 依赖列表
```

## 环境配置

创建 Python 3.11 环境并安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch 2.8.0
- Accelerate 1.2.1
- Diffusers 0.32.1
- Transformers 4.47.0
- timm 1.0.12
- xformers 0.0.32

### 下载预训练模型

**VA-VAE 模型下载：**

本项目使用 VA-VAE (Vision Foundation Model Aligned VAE) 进行图像编码和解码。需要下载预训练权重：

```bash
# 创建模型目录
mkdir -p checkpoints

# 下载 VA-VAE 模型权重
wget https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/vavae-imagenet256-f16d32-dinov2.pt \
    -O checkpoints/vavae-imagenet256-f16d32-dinov2.pt
```

或者手动下载：
- **下载地址**: https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/vavae-imagenet256-f16d32-dinov2.pt
- **放置位置**: 更新配置文件 `tokenizer/configs/vavae_f16d32_vfdinov2.yaml` 中的 `ckpt_path` 为实际路径

配置文件示例：

```yaml
ckpt_path: /path/to/checkpoints/vavae-imagenet256-f16d32-dinov2.pt

model:
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    embed_dim: 32
    use_vf: dinov2
    # ... 其他配置
```

## 数据集准备

本项目使用 **JsonLabelDataset** 数据集格式，支持文本描述作为条件标签。

### 数据集下载

推荐使用以下公开数据集进行训练：

- **text-to-image-2M**: https://huggingface.co/datasets/jackyhate/text-to-image-2M

该数据集包含约 200 万张图像及对应的文本描述，适合用于文本条件扩散模型的训练。

### 数据目录结构

```
data_dir/
├── images/           # 原始图像文件 (256x256 或 512x512)
│   ├── data_000000/
│   │   ├── flux_512_100k_00000014.png
│   │   ├── flux_512_100k_00000015.png
│   │   └── ...
│   └── data_000001/
│       └── ...
│
├── vae-in/           # VA-VAE 编码的潜在向量
│   ├── data_000000/
│   │   ├── img-latents-flux_512_100k_00000014.npy
│   │   ├── img-latents-flux_512_100k_00000015.npy
│   │   └── ...
│   ├── data_000001/
│   │   └── ...
│   └── latents_stats.pt    # 自动生成的统计文件 (mean/std)
│
└── labels.json       # 文本标签文件
```

### labels.json 格式说明

JSON 数组格式，每个元素包含 `id` 和 `en` 两个字段：

```json
[
    {"id": "data_000000/flux_512_100k_00000014", "en": "A fluffy cat sitting on a windowsill, sunlight streaming through"},
    {"id": "data_000000/flux_512_100k_00000015", "en": "A beautiful sunset over the ocean with orange and purple clouds"},
    {"id": "data_000001/flux_512_100k_00000100", "en": "A modern city skyline at night with glowing lights"}
]
```

**字段说明：**
- `id`: 文件标识符，对应图像/潜在向量的路径（不含扩展名）
- `en`: 英文文本描述，作为生成条件

### 文件 ID 映射规则

数据集会自动处理以下文件名格式的映射：

| 潜在向量文件名 | 提取的 ID |
|---------------|----------|
| `data_000000/img-latents-flux_512_100k_00000014.npy` | `data_000000/flux_512_100k_00000014` |
| `data_000000/flux_512_100k_00000014.npy` | `data_000000/flux_512_100k_00000014` |

图像文件和潜在向量文件需要一一对应，数量必须相同。

### 潜在向量统计

首次加载数据集时，会自动计算潜在向量的统计信息（均值和标准差），并缓存到 `vae-in/latents_stats.pt` 文件中。训练时会自动对潜在向量进行标准化：

```python
features = (features - mean) / std
```

### 数据预处理

#### 完整的数据准备流程

假设您有原始图像和对应的文本描述，完整的数据准备流程如下：

**1. 准备原始图像和标签**

```
my_dataset/
├── raw_images/          # 原始图像（任意尺寸）
│   ├── cat001.jpg
│   ├── cat002.jpg
│   └── ...
└── captions.json        # 文本标签
```

**2. 编码为 VA-VAE 潜在向量**

```bash
# 使用 encode_vavae 命令将图像编码为 VA-VAE 潜在向量
python preprocessing/dataset_tools.py encode-vavae \
    --config=tokenizer/configs/vavae_f16d32_vfdinov2.yaml \
    --source=my_dataset/raw_images \
    --dest=my_dataset/vae-in \
    --gpus=8 \
    --batch-size=100
```

**3. 整理最终数据集**

编码完成后，将图像复制到 `images` 目录，并创建 `labels.json`：

```bash
# 复制图像到标准目录（如果还没有的话）
cp -r my_dataset/raw_images my_dataset/images

# 确保 labels.json 在数据集根目录
cp my_dataset/captions.json my_dataset/labels.json
```

最终目录结构：

```
my_dataset/
├── images/              # 原始图像
│   ├── data_000000/
│   │   ├── cat001.jpg
│   │   └── cat002.jpg
│   └── ...
├── vae-in/             # VA-VAE 潜在向量
│   ├── data_000000/
│   │   ├── cat001.npy
│   │   └── cat002.npy
│   ├── latents_stats.pt  # 自动生成
│   └── ...
└── labels.json         # 文本标签
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--config` | VA-VAE 配置文件路径 | **必填** |
| `--source` | 输入图像目录 | **必填** |
| `--dest` | 输出潜在向量目录 | **必填** |
| `--gpus` | 并行编码使用的 GPU 数量 | 8 |
| `--batch-size` | 每个 GPU 每批处理的图像数 | 100 |
| `--max-images` | 最大处理图像数 | 无限制 |

**注意事项：**
- VA-VAE 编码不需要标签文件，只保存潜在向量
- 输出文件名会保留原始文件的目录结构和名称，便于与 labels.json 中的 ID 匹配
- 支持多 GPU 并行处理，图像以轮询方式分配到各 GPU
- 首次加载数据集时会自动计算潜在向量统计信息并缓存

### 使用自定义标签文件

可以通过 `--label-file` 参数指定标签文件路径（默认为 `data_dir/labels.json`）：

```bash
accelerate launch train.py \
    --use-json-dataset \
    --data-dir="/path/to/data_dir/" \
    --label-file="/path/to/custom_labels.json" \
    ...
```

## 训练

### 基础训练命令

```bash
accelerate launch train.py \
    --use-json-dataset \
    --model="SiT-B/1" \
    --data-dir="/path/to/data_dir/" \
    --label-file="/path/to/data_dir/labels.json" \
    --batch-size=256 \
    --learning-rate=2e-5 \
    --mixed-precision="bf16" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --cfm-coeff=0.05 \
    --cfm-weighting="uniform" \
    --time-shifting \
    --report-to="wandb" \
    --exp-name="omegadit-exp"
```

### 主要训练参数说明

| 参数 | 说明 | 默认值 | 推荐值 |
|-----|------|-------|-------|
| `--use-json-dataset` | 使用文本标签数据集 | False | **必须启用** |
| `--model` | 模型架构 | - | SiT-B/1 |
| `--data-dir` | 数据集根目录 | - | 包含 images、vae-in、labels.json |
| `--label-file` | 标签文件路径 | `data_dir/labels.json` | 可自定义 |
| `--batch-size` | 全局批大小（多卡总和） | 8 | 256 |
| `--learning-rate` | 学习率 | 2e-5 | 2e-5 |
| `--mixed-precision` | 混合精度模式 | fp16 | bf16 |
| `--enc-type` | DINOv2 特征提取器 | dinov2-vit-b | dinov2-vit-b/l |
| `--proj-coeff` | 投影损失系数 (α) | 0.5 | 0.5 |
| `--cfm-coeff` | CFM 损失系数 (β) | 0.05 | 0.05 |
| `--cfm-weighting` | CFM 损失加权方式 | uniform | uniform/linear |
| `--cfg-prob` | Classifier-free guidance 丢弃概率 | 0.1 | 0.1 |
| `--time-shifting` | 启用时间移位 | True | 推荐启用 |
| `--shift-base` | 时间移位基数 | 4096 | 4096 |
| `--max-train-steps` | 最大训练步数 | 400000 | 400000 |
| `--checkpointing-steps` | 检查点保存间隔 | 10000 | 10000 |
| `--clip-model-id` | CLIP 文本编码器 | AI-ModelScope/CLIP-GmP-ViT-L-14 | 默认 |

### CLIP 文本编码器

本项目使用 **ModelScope CLIP-GmP-ViT-L-14** 进行文本编码：

```python
# 自动从 ModelScope 下载并加载
clip_model_id = "AI-ModelScope/CLIP-GmP-ViT-L-14"
```

文本会被编码为 77 个 token 的嵌入向量，作为扩散模型的条件输入。

### 损失函数组成

```
L_total = L_denoise + α·L_proj + β·L_cfm

其中:
- L_denoise: 标准扩散去噪损失 (MSE)
- L_proj: DINOv2 特征投影对齐损失
- L_cfm: 对比流匹配损失

默认参数: α=0.5, β=0.05
```

### 恢复训练

从检查点恢复训练：

```bash
accelerate launch train.py \
    --use-json-dataset \
    --resume-step=100000 \
    --exp-name="omegadit-exp" \
    ...其他参数...
```

### 检查点保存

检查点保存位置：

```
exps/<exp-name>/checkpoints/<step>.pt
```

每个检查点包含：
- `model`: 模型权重
- `opt`: 优化器状态
- `args`: 训练参数
- `steps`: 当前训练步数

## 采样与评估

### 生成样本
- 修改eval.sh的相关参数和路径
- 修改generate.py 中 155 行 get_latent_stats 的路径参数
- 支持使用文本提示进行采样，在 `generate.py` 中可以设置硬编码的提示词。

```bash
sh eval.sh
```

### 计算评估指标

```bash
python evaluations/evaluator.py \
    /path/to/reference.npz \
    /path/to/samples.npz
```

评估指标包括：
- FID (Fréchet Inception Distance)
- sFID (Spatial FID)
- IS (Inception Score)
- Precision / Recall

## 模型架构

```
输入图像 (256x256)
    ↓
VA-VAE 编码 (DINOv2 对齐)
    ↓
潜在空间 (C=32, H=16, W=16)
    ↓
SiT-B/1 扩散模型 (140M 参数)
├─ TimestepEmbedder (时间步嵌入)
├─ CaptionEmbedder (条件嵌入: 类标签/CLIP文本)
├─ DiT Blocks
│  ├─ 融合自注意力 (QK 归一化)
│  ├─ 交叉注意力 (文本条件)
│  ├─ AdaLN-Zero 调制
│  └─ FFN (SiLU 激活)
├─ 投影器 (DINOv2 特征对齐)
└─ 输出头 (V 预测)
    ↓
VA-VAE 解码
    ↓
生成图像
```

## 与原项目的主要区别

| 特性 | 原 SpeedrunDiT | OmegaDiT |
|-----|---------------|-------------------|
| VAE | INVAE | VA-VAE (DINOv2 对齐) |
| 条件类型 | ImageNet 类标签 | 类标签 + 文本描述 |
| 文本编码器 | 无 | CLIP-GmP-ViT-L-14 |
| 数据集格式 | 固定类标签 | 支持 JSON 文本标签 |

---

## Omega-AI: Java 深度学习框架实现

**对于 Java 开发者，本模型已在 [Omega-AI](https://gitee.com/dromara/omega-ai) 框架中完整实现！**

### 关于 Omega-AI

[Omega-AI](https://gitee.com/dromara/omega-ai) 是由 Dromara 社区开源的一款**基于 Java 打造的深度学习框架**，旨在帮助 Java 开发者快速搭建神经网络，实现模型推理与训练。

### 核心特性

- **纯 Java 实现**: 无需 Python 环境，Java 开发者可直接使用
- **自动求导引擎**: 支持自动微分，简化梯度计算
- **多 GPU 训练**: 支持分布式多卡训练，加速模型收敛
- **CUDA/CUDNN 加速**: GPU 计算支持 NVIDIA CUDA 和 CUDNN 加速
- **完整的模型支持**: 已实现 OmegaDiT 扩散模型，可直接进行推理和训练

### 项目链接

- **Gitee 仓库**: https://gitee.com/dromara/omega-ai
- **Dromara 社区**: https://dromara.org/

### 为什么选择 Omega-AI？

| 对比项 | Python (PyTorch) | Java (Omega-AI) |
|-------|------------------|-----------------|
| 语言生态 | Python | Java/JVM |
| 部署环境 | 需要 Python 运行时 | JVM 即可运行 |
| 企业集成 | 需要额外封装 | 原生 Java，无缝集成 |
| 微服务部署 | 较复杂 | Spring Boot 等框架直接集成 |
| 性能 | 优秀 | GPU 加速，性能优秀 |

如果你是 Java 开发者，或者你的项目主要使用 Java 技术栈，**强烈推荐使用 Omega-AI** 来运行 OmegaDiT 模型，享受纯 Java 环境下的深度学习体验！

---

## 致谢

本项目基于 [SpeedrunDiT](https://github.com/SwayStar123/SpeedrunDiT/) 开发，感谢原作者 Swayam Bhanded 的开源贡献。

同时感谢以下项目和资源：

- [SpeedrunDiT](https://github.com/SwayStar123/SpeedrunDiT/) - 原始项目
- [REG / REPA](https://github.com/SwayStar123/REG) - 表示对齐方法
- [SiT](https://github.com/willisma/SiT) - Scalable Interpolant Transformers
- [DINOv2](https://github.com/facebookresearch/dinov2) - 视觉基础模型
- [VA-VAE](https://github.com/hustvl/LightningDiT/tree/main/vavae) - Vision Foundation Model Aligned VAE (作者: Maple/Jingfeng Yao)
- [ADM Evaluations](https://github.com/openai/guided-diffusion) - 评估代码
- [NVLabs edm2](https://github.com/NVlabs/edm2) - 预处理工具

特别感谢 **[Omega-AI](https://gitee.com/dromara/omega-ai)** 项目和 **Dromara 开源社区**，为 Java 开发者提供了本模型的完整实现，让更多开发者能够在 Java 生态中使用扩散模型技术。

## 引用

如果您使用了本项目，请引用以下相关论文：

**SR-DiT (SpeedrunDiT)**: https://arxiv.org/abs/2512.12386

```bibtex
@misc{bhanded2025speedrundit,
  title         = {Speedrunning ImageNet Diffusion},
  author        = {Bhanded, Swayam},
  year          = {2025},
  eprint        = {2512.12386},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2512.12386},
}
```

**SPRINT**: https://arxiv.org/abs/2510.21986

```bibtex
@misc{mukherjee2025sprint,
  title         = {Sprint: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers},
  author        = {Dogyun Park, Moayed Haji-Ali,Yanyu Li ,Willi Menapace, Sergey Tulyakov, Hyunwoo J. Kim, Aliaksandr Siarohin, Anil Kag},
  year          = {2025},
  eprint        = {2510.21986},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2510.21986},
}
```

## 联系方式

如有问题，请提交 GitHub Issue。
