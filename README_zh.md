# Distinguishing Visually Similar Actions: Prompt-Guided Semantic Prototype Modulation for Few-Shot Action Recognition

## 项目概述

### 研究背景
在少样本动作识别领域，区分视觉相似的动作是一个关键挑战。现有的方法往往依赖于视觉特征的相似性匹配，而忽略了动作的语义信息，导致在处理视觉相似但语义不同的动作时性能下降。为了解决这一问题，我们提出了一种基于提示引导的语义原型调制方法，旨在利用语义信息增强视觉特征的区分能力。

### 创新点
1. **CLIP-SPM跨模态框架**：提出了CLIP-SPM跨模态框架，联合解决少样本动作识别的三个核心挑战：鲁棒的时间建模、细粒度视觉相似性和支持-查询模态鸿沟。
2. **分层协同运动细化（HSMR）模块**：设计了分层协同运动细化模块，显式提取浅层和深层运动特征，在稀缺监督下通过层级一致性正则化强制捕获稳定的运动模式。
3. **语义原型调制（SPM）策略**：开发了语义原型调制策略，从Episode上下文中学习查询相关的文本提示，以弥合模态鸿沟。SPM通过利用文本特征的语义指导来调制视觉特征，增强视觉相似动作之间的可区分性。
4. **原型-锚点双调制（PADM）方法**：引入了原型-锚点双调制方法，联合细化支持侧原型并将查询特征对齐到全局锚点，产生更强的跨集一致性和更具判别性的原型。

### 主要贡献
1. 提出了一种新的少样本动作识别框架，有效区分视觉相似的动作
2. 设计了提示引导的语义原型调制机制，实现了语义信息和视觉信息的深度融合
3. 在多个基准数据集上取得了优异的性能，证明了方法的有效性和泛化能力

## 架构图

![CLIP-SPM Architecture](./imgs/architecture.png)

## 实验结果

### 数据集信息
我们在五个广泛使用的少样本动作识别基准上评估了我们的模型：

| 数据集 | 总类别数 | 训练/验证/测试类别数 | 训练视频数 | 验证视频数 | 测试视频数 | 总视频数 | 分割方式 | 特殊说明 |
|--------|----------|----------------------|------------|------------|------------|----------|----------|----------|
| Kinetics   | 100 | 64/12/24 | 6,400 | 1,200 | 2,400 | 10,000 | CMN | 每个类别100个样本 |
| SSv2-Full  | 174 | 64/12/24 | 77,500 | 1,926 | 2,854 | 82,280 | OTAM | 使用所有可用样本 |
| SSv2-Small | 100 | 64/12/24 | 6,400 | 1,200 | 2,400 | 10,000 | CMN-J | 每个类别100个样本 |
| UCF101     | 101 | 70/10/21 | 9,154 | 1,421 | 2,745 | 13,320 | ARN | - |
| HMDB51     | 51  | 31/10/10 | 4,280 | 1,194 | 1,292 | 6,766 | ARN | - |

### 对比实验结果

我们在五个基准数据集上对CLIP-SPM模型进行了全面评估，使用CLIP-RN50和CLIP-ViTB16作为基础模型。以下是在不同shot设置下的性能对比：

#### CLIP-RN50 基础模型
| 数据集       | 1-shot | 3-shot | 5-shot |
|--------------|--------|--------|--------|
| HMDB51       | 77.4   | 82.6   | 84.5   |
| UCF101       | 93.8   | 97.2   | 97.7   |
| Kinetics     | 90.9   | 92.9   | 93.5   |
| SSv2-Full    | 62.3   | 67.0   | 69.4   |
| SSv2-Small   | 50.8   | 54.9   | 58.5   |

#### CLIP-ViTB16 基础模型
| 数据集       | 1-shot | 3-shot | 5-shot |
|--------------|--------|--------|--------|
| HMDB51       | 78.2   | 86.3   | 88.6   |
| UCF101       | 96.2   | 98.2   | 98.7   |
| Kinetics     | 92.8   | 94.1   | 94.3   |
| SSv2-Full    | 66.7   | 74.8   | 77.3   |
| SSv2-Small   | 57.8   | 62.4   | 66.2   |

**注**：表中所有结果均为CLIP-SPM模型在各数据集上的性能表现，数据来源为实验结果统计。

## 项目结构说明

```
CLIP-SPM/
├── configs/                  # 配置文件
│   ├── clipspm/              # CLIP-SPM模型配置
│   ├── clipfsar/             # CLIP-FSAR模型配置
│   ├── cpm2c/                # CPM2C模型配置
│   ├── otam/                 # OTAM模型配置
│   └── base.yaml             # 基础配置
├── models/                   # 模型定义
│   ├── model_clipspm.py      # CLIP-SPM模型实现
│   ├── model_clipfsar.py     # CLIP-FSAR模型实现
│   ├── model_cpm2c.py        # CPM2C模型实现
│   ├── model_otam.py         # OTAM模型实现
│   └── clip_fsar.py          # CLIP基础模型
├── run/                      # 运行脚本
│   ├── run.py                # 主运行脚本
│   └── main_run.py           # 训练和测试逻辑
├── splits/                   # 数据集分割文件
│   ├── hmdb_ARN/             # HMDB51数据集分割
│   ├── ssv2_CMN/             # SSv2-CMN数据集分割
│   ├── ssv2_OTAM/            # SSv2-OTAM数据集分割
│   └── ucf_ARN/              # UCF101数据集分割
├── utils/                    # 工具函数
│   ├── config.py             # 配置文件处理
│   └── utils.py              # 通用工具函数
├── videotransforms/          # 视频变换
│   ├── video_transforms.py   # 视频空间变换
│   └── volume_transforms.py  # 视频时间变换
├── requirements.txt          # 依赖项列表
├── train_test.sh             # 训练测试脚本
└── README.md                 # 项目说明文档
```

## 环境配置指南

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-username/CLIP-SPM.git
   cd CLIP-SPM
   ```

2. **创建虚拟环境**
   ```bash
   conda create -n clipspm python=3.8
   conda activate clipspm
   ```

3. **安装PyTorch和CUDA**
   ```bash
   conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

4. **安装其他依赖项**
   ```bash
   pip install -r requirements.txt
   ```

5. **安装CLIP模型**
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

## 使用教程

### 数据准备

我们采用了[CLIP-MEI](https://github.com/D-XH/CLIP-MEI)提供的数据集

### 训练脚本使用方法

   ```bash
   sh train_test.sh
   ```

### 测试脚本使用方法

当想要测试时，修改配置文件的ONLY_TEST为true，然后执行
   ```bash
   sh train_test.sh
   ```

### 预训练模型下载链接

百度云盘：[Checkpoints](https://pan.baidu.com/s/1WzV8uXlbYu_bWe-xvaYafg?pwd=284q) 提取码：284q

## 引用信息

如果您使用了本项目的代码或模型，请引用我们的论文：

```latex
@article{li2025_2512.19036,
  title={Distinguishing Visually Similar Actions: Prompt-Guided Semantic Prototype Modulation for Few-Shot Action Recognition},
  author={Xiaoyang Li and Mingming Lu and Ruiqi Wang and Hao Li and Zewei Le},
  journal={arXiv preprint arXiv:2512.19036},
  year={2025}
}
```

同时，我们的工作也受到了以下论文的启发，在此表示感谢：

```latex
@article{deng2025clip,
   title={CLIP-MEI: Exploit more effective information for few-shot action recognition},
   author={Deng, XuanHan and Yang, WenZhu and Zhao, XinBo and Zhou, Tong and Deng, Xin},
   journal={Knowledge-Based Systems},
   pages={113965},
   year={2025},
   publisher={Elsevier}
}
```

## 许可证

该项目遵循[Apache 2.0 许可协议](http://www.apache.org/licenses/LICENSE-2.0)的条款进行授权。

## 更新日志

### 2025-12-12
- 首次发布项目的 README.md
