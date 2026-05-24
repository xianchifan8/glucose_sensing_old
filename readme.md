# 🩸 Glucose Sensing Training Framework

> 基于深度学习的无创血糖监测训练框架 - 通过分析频谱数据实时预测血糖值

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9+](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [模型架构](#模型架构)
- [数据处理](#数据处理)
- [训练模式](#训练模式)
- [命令行参数](#命令行参数)
- [使用示例](#使用示例)
- [数据融合](#数据融合)
- [自回归模式](#自回归模式)
- [进阶功能](#进阶功能)
- [实验追踪](#实验追踪)
- [性能优化](#性能优化)
- [常见问题](#常见问题)
- [开发指南](#开发指南)
- [相关文档](#相关文档)

---

## 项目简介

本项目实现了一个**模块化、可扩展的深度学习训练框架**，用于基于频谱数据的血糖预测任务。支持多种模型架构、灵活的数据处理策略和完整的实验追踪系统。

### 🎯 核心特性

#### 📊 多样化模型支持
- ✅ **Instant模式**：MLP、CNN、Transformer
- ✅ **Window模式**：Transformer、**TCN（Temporal Convolutional Network）**
- ✅ **自动模型选择**：根据数据模式自动适配

#### 🔬 灵活数据处理
- ✅ **4种划分策略**：random（随机）、temporal（时序）、stratified（分层）、experiment（按实验）
- ✅ **两种预测模式**：instant（单时刻）、window（时序窗口）
- ✅ **数据融合**：可选的辅助传感器数据融合（BME680、PPG、T117、ICM）
- ✅ **自回归模式**：使用历史血糖增强预测稳定性（平滑性正则化）
- ✅ **自动归一化**：防止数据泄漏的标准化流程

#### 🎛️ 训练配置
- ✅ **可选验证集**：支持有/无验证集两种模式
- ✅ **实时监控**：每epoch评估测试集，及时发现过拟合
- ✅ **多种损失函数**：MSE、MAE、Huber Loss（支持AR复合损失）
- ✅ **自适应学习率**：CosineAnnealing、ReduceLROnPlateau、StepLR
- ✅ **正则化**：Dropout、Weight Decay、Gradient Clipping

#### 📈 实验管理
- ✅ **W&B集成**：完整的实验追踪和可视化
- ✅ **自动命名**：智能的run命名规则
- ✅ **性能分析**：详细的计时统计
- ✅ **NaN检测**：自动跳过异常batch

### 🛠️ 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 深度学习 | PyTorch | 2.9.1+ |
| 数据处理 | NumPy, Pandas | Latest |
| 机器学习 | scikit-learn | Latest |
| 实验追踪 | Weights & Biases | Latest |
| 可视化 | Matplotlib, Seaborn | Latest |
| 环境管理 | Conda/Pip | - |

---

## 项目结构

```
Training/
│
├── 📄 main.py                          # 主程序入口
├── ⚙️ config.py                        # 配置管理（数据类）
├── 🏋️ trainer.py                       # 训练器实现
├── 🔧 utils.py                         # 工具函数
├── 📖 readme.md                        # 项目文档（本文件）
│
├── 🧠 Model/                           # 模型定义目录
│   ├── __init__.py                     # 模块导出
│   ├── models.py                       # Instant模式模型（MLP、CNN、Transformer）
│   └── window/                         # Window模式模型
│       ├── __init__.py
│       ├── models.py                   # Window LSTM、Transformer
│       └── tcn_models.py              # TCN模型（4层，感受野31步）
│
├── 📊 data/                            # 数据处理目录
│   ├── __init__.py
│   ├── data_loader.py                 # 数据加载器（频谱+血糖+辅助传感器）
│   └── dataset.py                     # 数据集处理、划分、窗口创建
│
├── 📈 visualization/                   # 可视化模块
│   ├── metrics.py                     # 指标计算（MAE、RMSE、MAPE）
│   └── plots.py                       # 绘图函数
│
├── 🧪 tests/                           # 单元测试
│   └── test_data_loader.py           # 数据加载测试
│
├── 💾 Dataset/                         # 数据目录
│   ├── Tao/                           # BIN格式用户数据
│   │   ├── 11141404_Tao/             # 实验1（自动解析时间）
│   │   │   ├── s000.bin, s001.bin    # 频谱数据
│   │   │   ├── a000.bin, a001.bin    # 辅助传感器数据（可选）
│   │   │   └── 11141404.json         # 血糖标签
│   │   ├── 11171040_Tao/             # 实验2
│   │   └── ...                        # 更多实验
│   │
│   └── Tao_db/                        # DB格式用户数据 ⭐ NEW
│       ├── experiments_config.json   # 实验配置文件
│       ├── qiangtao_glucose.db       # 血糖数据库 (blood_sugar表)
│       ├── DEV_002_qiangtao_260121/  # 实验1
│       │   └── DEV_002_qiangtao_260121.db  # 频谱数据库 (spectrum表)
│       └── DEV_002_qiangtao_260123/  # 实验2
│           └── DEV_002_qiangtao_260123.db
│
├── 💼 checkpoints/                     # 模型保存
├── 📊 results/                         # 可视化结果
├── 📝 wandb/                          # W&B日志
│
└── 📚 文档/                            # 详细文档
    ├── DATA_FUSION_USAGE.md          # 数据融合指南 ⭐ NEW
    ├── DATASET_SELECTION_GUIDE.md     # 数据集选择
    ├── NO_VAL_MODE_GUIDE.md          # 无验证集模式
    ├── WINDOW_MODE_GUIDE.md          # Window模式训练
    ├── TCN_ARCHITECTURE.md           # TCN架构详解 ⭐ NEW
    ├── TCN_GUIDE.md                  # TCN使用指南 ⭐ NEW
    ├── TCN_VS_OTHERS.md              # TCN性能对比 ⭐ NEW
    ├── TRANSFORMER_OPTIMIZATION.md    # Transformer优化
    ├── OVERFITTING_SOLUTIONS.md      # 过拟合解决方案
    └── QUICKSTART_WINDOW.md          # Window模式快速开始
```

---

## 快速开始

### 环境设置

1. **创建Conda环境**
```bash
conda create -n glucose_sensing python=3.12
conda activate glucose_sensing
```

2. **安装依赖**
```bash
# PyTorch (CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 其他依赖
pip install wandb numpy pandas scikit-learn matplotlib seaborn tqdm scipy
```

3. **登录W&B**
```bash
wandb login
```

### 基础训练

```bash
# 最简单的训练（MLP, instant模式）
python main.py --epochs 30 --normalize

# Window模式训练（推荐用于时序预测）
python main.py --model TCN --mode window --window_size 50 --epochs 30 --normalize

# 使用数据融合（辅助传感器）
python main.py --model TCN --mode window --window_size 50 --data_fusion --normalize
```

### 查看结果

- **终端输出**：实时显示训练进度
- **W&B Dashboard**：https://wandb.ai/glucose_msra/sensing
- **可视化结果**：`results/` 目录

---

## 模型架构

### 📌 Instant模式模型

#### 1. MLP (Multi-Layer Perceptron)
```python
输入(1001) → Linear(1001→256) + BN + ReLU + Dropout(0.5)
          → Linear(256→128) + BN + ReLU + Dropout(0.5)
          → Linear(128→1)
```
- **参数量**: ~290K
- **速度**: 最快
- **适用**: 快速实验、基线模型

#### 2. CNN (1D Convolutional Neural Network)
```python
输入(1001) → Conv1d(1→32, k=5) + BN + ReLU + MaxPool(2)
          → Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)
          → Conv1d(64→128, k=3) + BN + ReLU + MaxPool(2)
          → Flatten → Linear(16000→256) + Dropout
          → Linear(256→1)
```
- **参数量**: ~2M
- **速度**: 快
- **适用**: 捕捉局部频谱特征

#### 3. Transformer
```python
输入(1001) → Embedding(1001→256)
          → Positional Encoding
          → TransformerEncoder (8 heads, 3 layers, dropout=0.1)
          → Mean Pooling
          → Linear(256→128) + Dropout
          → Linear(128→1)
```
- **参数量**: ~3M
- **速度**: 慢
- **适用**: 全局注意力建模
- **优化**: 已优化dropout和初始化

### 📌 Window模式模型

#### 4. LSTM (Long Short-Term Memory)
```python
输入(window_size, 1001) → LSTM(1001→256, 2 layers, dropout=0.5)
                        → 取最后时间步输出
                        → Linear(256→128) + Dropout
                        → Linear(128→1)
```
- **参数量**: ~800K
- **速度**: 中等
- **适用**: 基础时序建模

#### 5. TCN (Temporal Convolutional Network) ⭐ **推荐**
```python
输入(window_size, 1001) 
  → SpectralCNN (多尺度特征提取)
     → conv_small (k=3): 捕捉相邻频点
     → conv_medium (k=7): 捕捉中等模式
     → conv_large (k=15): 捕捉全局趋势
     → 输出: (window_size, 128)
  
  → TCN (时序建模, 4层)
     → Layer 1 (dilation=1, 感受野=3步)
     → Layer 2 (dilation=2, 感受野=7步)
     → Layer 3 (dilation=4, 感受野=15步)
     → Layer 4 (dilation=8, 感受野=31步)
     → 输出: (256,)
  
  → 全连接层
     → Linear(256→128) + BN + ReLU
     → Linear(128→1)
```
- **参数量**: ~1.5M
- **速度**: 快（比LSTM快3-4倍）
- **感受野**: 31个时间步（约15.5分钟）
- **适用**: **时序预测的首选模型**
- **详见**: [TCN_ARCHITECTURE.md](TCN_ARCHITECTURE.md)

---

## 数据处理

### 数据格式

**频谱数据** (`s*.bin`):
- 格式: 二进制文件
- 内容: timestamp + 1001维功率谱 (power_0 到 power_1000)
- 采样: ~10Hz

**血糖标签** (两种格式支持):

*方式1: JSON文件* (`*.json`):
- 格式: JSON数组
- 内容: `[{"time": "MM.DD HH:MM", "value": float}, ...]`
- 采样: 不规则

*方式2: SQLite数据库* (`*_glucose.db`) ⭐ NEW:
- 表名: `blood_sugar`
- 字段: `create_time` (Unix时间戳UTC), `blood_sugar` (mmol/L)
- 优先级: glucose_db_path > json_path

**辅助传感器** (`a*.bin`, 可选):
- BME680: 温度、气压、湿度
- PPG: 红光、红外光强度
- T117: 高精度温度
- ICM-20948: 3轴加速度 + 3轴陀螺仪

### 数据流程

```
原始数据 
  → 加载实验 (自动解析目录名时间)
  → 时间对齐 (频谱 ↔ 血糖)
  → 插值血糖 (为每个频谱时间点)
  → [可选] 加载辅助传感器
  → [可选] 插值传感器到频谱时间戳
  → [可选] 特征融合 (spectrum + aux)
  → 划分数据集 (train/val/test)
  → [可选] 创建时间窗口 (window模式)
  → 归一化 (StandardScaler)
  → DataLoader
  → 训练
```

### 归一化策略

```python
# 方法1: 统一归一化（默认）
scaler = StandardScaler()
scaler.fit(train_data)  # 只在训练集上fit
train_data = scaler.transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# 方法2: 分组归一化（用于数据融合）
# 频谱特征和辅助传感器特征分别归一化
# 避免1001维 vs 12维的不平衡问题
```

**重要**: 先划分后归一化，避免数据泄漏！

---

## 训练模式

### 🔸 Instant模式
```bash
python main.py --model CNN --epochs 50 --normalize
```
- **输入**: 单个时刻的频谱 (1001维)
- **输出**: 该时刻的血糖值
- **适用**: 快速训练、模型对比
- **模型**: MLP, CNN, RNN, Transformer

### 🔹 Window模式  
```bash
python main.py --model TCN --mode window --window_size 50 --epochs 30 --normalize
```
- **输入**: 窗口内多个时刻的频谱序列 `(window_size, 1001)`
- **输出**: 当前时刻的血糖值
- **优势**: 利用历史信息，捕捉血糖变化趋势
- **模型**: LSTM, Transformer, **TCN（推荐）**
- **详见**: [WINDOW_MODE_GUIDE.md](WINDOW_MODE_GUIDE.md)

### 窗口大小选择

| Window Size | 时长 (30秒采样) | 适用场景 |
|-------------|----------------|---------|
| 10 | 5分钟 | 快速测试 |
| 30 | 15分钟 | 短期趋势 |
| 50 | 25分钟 | 标准配置 ⭐ |
| 100 | 50分钟 | 长期趋势 |
| 200 | 100分钟 | 完整进食周期（慎用，内存占用大）|

---

## 命令行参数

### 训练参数
```bash
--lr FLOAT              # 学习率（默认：0.001）
--epochs INT            # 训练轮数（默认：50）
--batch_size INT        # 批次大小（默认：32）
--seed INT              # 随机种子（默认：42）
--weight_decay FLOAT    # 权重衰减（默认：1e-4）⭐ NEW
--loss {MSE,MAE,Huber}  # 损失函数
--huber_delta FLOAT     # Huber Loss的delta参数
```

### 模型参数
```bash
--model {MLP,CNN,RNN,LSTM,Transformer,TCN,SIMPLE_TCN}
--hidden_size INT       # 隐藏层大小（默认：256）
--num_layers INT        # 层数（默认：4）⭐ TCN默认4层
--dropout FLOAT         # Dropout率（默认：0.5，TCN推荐0.4）
```

### 数据参数
```bash
--normalize                    # 启用归一化
--data_fusion                  # 启用辅助传感器数据融合
--split_strategy {random,temporal,stratified,experiment}
--mode {instant,window}        # 预测模式
--window_size INT              # 窗口大小（默认：10）

--experiments EXP1 EXP2 ...    # 指定使用的实验
--user_experiments USER EXP1 EXP2  # 多用户实验映射 ⭐ NEW
--train_split FLOAT            # 训练集比例
--val_split FLOAT              # 验证集比例
--test_split FLOAT             # 测试集比例
--no_val                       # 不使用验证集

# 数据源参数 ⭐ NEW
--data_source {bin,db}         # 数据源格式（默认: bin）
--user USER                    # BIN格式单用户
--users USER1 USER2 ...        # BIN格式多用户
--db_user USER_DB              # DB格式单用户（默认: Tao_db）
--db_users USER_DB1 USER_DB2   # DB格式多用户
```

### W&B参数
```bash
--no_wandb              # 禁用W&B
--wandb_name NAME       # 自定义run名称
```

---

## 使用示例

### 基础训练

```bash
# Instant模式：快速基线
python main.py --model MLP --epochs 30 --normalize

# Window模式：TCN（推荐）
python main.py --model TCN --mode window --window_size 50 --epochs 30 --normalize

# 完整配置
python main.py \
  --model TCN \
  --mode window \
  --window_size 50 \
  --epochs 30 \
  --normalize \
  --split_strategy temporal \
  --no_val \
  --train_split 0.5 \
  --test_split 0.5 \
  --dropout 0.4 \
  --num_layers 4 \
  --weight_decay 0.001
```

### 数据融合训练

```bash
# 启用辅助传感器数据
python main.py \
  --model TCN \
  --mode window \
  --window_size 50 \
  --data_fusion \
  --normalize \
  --epochs 30
```

**特征维度变化**:
- 不使用融合: 1001维（仅频谱）
- 使用融合: 1001+12=1013维（频谱+辅助传感器）

**详见**: [DATA_FUSION_USAGE.md](DATA_FUSION_USAGE.md)

### DB数据源训练 ⭐ NEW

```bash
# DB格式单用户训练（血糖从glucose.db读取）
python main.py \
  --data_source db \
  --db_user Tao_db \
  --model TCN \
  --mode window \
  --window_size 50 \
  --normalize \
  --epochs 30

# DB格式多用户训练
python main.py \
  --data_source db \
  --db_users Tao_db Weiyi_db \
  --model TCN \
  --mode window \
  --normalize \
  --epochs 30

# 多用户分别指定实验 ⭐ NEW
python main.py \
  --data_source db \
  --db_users Tao_db Weiyi_db \
  --user_experiments Tao_db 260121_1703_Tao 260123_1257_Tao \
  --user_experiments Weiyi_db 260122_1000_Weiyi \
  --model TCN \
  --mode window \
  --normalize \
  --epochs 30
```

**DB格式配置文件** (`experiments_config.json`):
```json
[
    {
        "db_path": "DEV_002_qiangtao_260121/DEV_002_qiangtao_260121.db",
        "glucose_db_path": "qiangtao_glucose.db",
        "start_time": "2026-01-21 17:03:00",
        "end_time": "2026-01-21 20:21:00",
        "experiment_name": "260121_1703_Tao"
    }
]
```

### 数据集选择

```bash
# 查看可用实验
ls Dataset/Tao/

# 使用特定实验
python main.py --experiments 11201828_Tao 11211544_Tao --epochs 30

# 使用所有实验（默认）
python main.py --epochs 30
```

### 超参数搜索

```bash
# 学习率搜索
for lr in 0.0001 0.001 0.01; do
  python main.py --model TCN --mode window --window_size 50 --lr $lr --epochs 30 --normalize
done

# 窗口大小搜索
for ws in 30 50 100; do
  python main.py --model TCN --mode window --window_size $ws --epochs 30 --normalize
done

# Dropout搜索
for drop in 0.3 0.4 0.5; do
  python main.py --model TCN --mode window --window_size 50 --dropout $drop --epochs 30 --normalize
done
```

---

## 数据融合

### 功能说明

将辅助传感器数据与频谱数据融合，提供更丰富的特征：

**支持的传感器**:
- BME680: 温度、气压、湿度 (3维)
- PPG: 红光、红外光强度 (2维)
- T117: 高精度温度 (1维)
- ICM-20948: 3轴加速度 + 3轴陀螺仪 (6维)
- **总计**: 最多12维额外特征

### 使用方法

```bash
# 基础用法
python main.py --data_fusion --normalize

# 完整配置
python main.py \
  --model TCN \
  --mode window \
  --window_size 50 \
  --data_fusion \
  --normalize \
  --epochs 30
```

### 数据融合流程

```
Step 1: 加载频谱数据 (S*.bin)
  ↓
Step 2: 加载辅助传感器 (A*.bin)
  ↓
Step 3: 插值对齐（线性插值到频谱时间戳）
  ↓
Step 4: 特征拼接
  spectrum_features (N, 1001) + aux_features (N, 12) → features (N, 1013)
  ↓
Step 5: 训练
```

**详见**: [DATA_FUSION_USAGE.md](DATA_FUSION_USAGE.md)

---

## 自回归模式

自回归（Auto-Regressive）模式通过引入历史血糖数据增强预测的时序连续性和稳定性。

### 核心特性

- ✅ **历史血糖作为输入**：模型同时接收频谱数据和历史血糖序列
- ✅ **平滑性正则化**：限制预测输出的跳变幅度
- ✅ **低耦合设计**：作为Wrapper包装基础模型，不影响TCN架构

### 快速使用

```bash
# 启用AR模式
python main.py --mode window --model tcn --autoregressive

# 自定义AR参数
python main.py --mode window --model tcn \
    --autoregressive \
    --ar_glucose_history 8 \
    --ar_smoothness_weight 0.15

# AR + Late Fusion（推荐组合）
python main.py --mode window --model tcn \
    --data_fusion --fusion_stage late \
    --autoregressive
```

### AR参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--autoregressive` | False | 启用自回归模式 |
| `--ar_glucose_history` | 5 | 使用多少个历史血糖值 |
| `--ar_aligned` | True | 历史不包含当前血糖（默认） |
| `--ar_smoothness_weight` | 0.1 | 平滑性正则化权重 |
| `--ar_history_weight` | 0.05 | 历史趋势一致性权重 |

**详见**: [AUTOREGRESSIVE_MODE.md](AUTOREGRESSIVE_MODE.md)

---

## 进阶功能

### 1. 数据划分策略

```bash
# 随机划分（默认）- 标准训练
python main.py --split_strategy random

# 时序划分 - 模拟真实场景（早期→训练，晚期→测试）
python main.py --split_strategy temporal

# 分层划分 - 保持血糖分布
python main.py --split_strategy stratified

# 按实验划分 - 整个实验作为一组
python main.py --split_strategy experiment
```

### 2. 无验证集模式

```bash
# 标准模式（70/15/15）
python main.py --epochs 30

# 无验证集模式（80/20）
python main.py --no_val --epochs 30

# 自定义比例（50/50，适合时序划分）
python main.py --no_val --train_split 0.5 --test_split 0.5 --split_strategy temporal
```

**详见**: [NO_VAL_MODE_GUIDE.md](NO_VAL_MODE_GUIDE.md)

### 3. TCN深度学习

**TCN架构优势**:
- ✅ 并行计算，比LSTM快3-4倍
- ✅ 固定感受野，梯度路径短
- ✅ 因果卷积，只看历史信息
- ✅ 多层dilated conv，捕捉不同时间尺度

```bash
# 标准TCN (4层, 感受野31步)
python main.py --model TCN --mode window --window_size 50 --num_layers 4

# 深层TCN (5层, 感受野63步)
python main.py --model TCN --mode window --window_size 100 --num_layers 5

# 浅层TCN (3层, 感受野15步，更快)
python main.py --model TCN --mode window --window_size 30 --num_layers 3
```

**详见**: [TCN_GUIDE.md](TCN_GUIDE.md), [TCN_ARCHITECTURE.md](TCN_ARCHITECTURE.md)

---

## 实验追踪

### W&B配置

- **Entity**: `glucose_msra`
- **Project**: `sensing`
- **Dashboard**: https://wandb.ai/glucose_msra/sensing

### Run命名规则

自动生成格式：
```
{模型}_{学习率}_ep{epochs}_split_{策略}_validate_{yes/no}_{归一化}_exps{数据集}
```

**示例**:
```
TCN_lr0.001_ep30_split_temporal_validate_no_normalized_exps_all
CNN_lr0.001_ep50_split_random_validate_yes_raw_exps3
```

### 记录的指标

**每个epoch记录**:
- `train_loss`, `train_mae`
- `val_loss`, `val_mae`, `val_rmse` (如果有验证集)
- `test_loss`, `test_mae`, `test_rmse` (实时监控)
- `learning_rate`

**模型信息**:
- 参数量
- 模型架构
- 超参数配置

---

## 性能优化

### GPU加速

```bash
# 检查GPU
python -c "import torch; print(torch.cuda.is_available())"

# 监控GPU使用
watch -n 1 nvidia-smi
```

### 训练速度对比

| 模型 | 相对速度 | 推荐batch_size | 推荐场景 |
|------|---------|---------------|---------|
| MLP | 1x (基准) | 32-64 | 快速实验 |
| CNN | 1.5x | 32-64 | Instant模式 |
| LSTM | 3x | 32 | Window基线 |
| **TCN** | **2x** | **32-64** | **Window首选** ⭐ |
| Transformer | 5x | 64-128 | 全局建模 |

### 加速建议

1. **增大batch_size**: `--batch_size 64`
2. **减小模型**: `--hidden_size 128 --num_layers 2`
3. **减小窗口**: `--window_size 30`
4. **使用TCN**: 比LSTM快3-4倍
5. **禁用W&B**: `--no_wandb`（测试时）

---

## 常见问题

### Q1: 训练时出现NaN？

**原因**:
- 梯度爆炸
- 学习率过大
- 数据未归一化

**解决方案**:
```bash
# 1. 启用归一化
python main.py --normalize

# 2. 降低学习率
python main.py --lr 0.0001 --normalize

# 3. 使用稳定模型
python main.py --model TCN --mode window --window_size 50 --normalize
```

### Q2: 内存不足 (OOM)？

**原因**:
- window_size太大
- batch_size太大
- 模型太大

**解决方案**:
```bash
# 减小窗口
python main.py --mode window --window_size 30

# 减小批次
python main.py --batch_size 16

# 减小模型
python main.py --hidden_size 128 --num_layers 2
```

### Q3: TCN vs LSTM vs Transformer？

| 指标 | LSTM | TCN ⭐ | Transformer |
|------|------|-------|-------------|
| 速度 | 中等 | 快 | 慢 |
| 感受野 | 全部 | 31步 | 全部 |
| 训练稳定性 | 好 | 很好 | 需调参 |
| 参数量 | 800K | 1.5M | 3M |
| **推荐度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**详见**: [TCN_VS_OTHERS.md](TCN_VS_OTHERS.md)

### Q4: 如何判断过拟合？

**标志**:
- ✓ 正常：Train MAE ≈ Test MAE
- ⚠️ 轻微：Test MAE > Train MAE 约10-20%
- ❌ 严重：Test MAE >> Train MAE 超过50%

**解决方案**: 见 [OVERFITTING_SOLUTIONS.md](OVERFITTING_SOLUTIONS.md)

### Q5: 数据融合有用吗？

**可能有帮助的场景**:
- 环境因素影响测量（温度、湿度）
- 运动状态影响血糖（加速度、陀螺仪）
- 心率/血氧与代谢相关（PPG）

**建议**: 先在小数据集上A/B测试，对比有无融合的效果

---

## 开发指南

### 添加新模型

1. **在 `Model/models.py` 或 `Model/window/` 定义模型**:
```python
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, ...):
        super(MyModel, self).__init__()
        # 定义层
        
    def forward(self, x):
        # 前向传播
        return x
```

2. **在 `create_model()` 或 `create_window_model()` 注册**:
```python
elif arch == 'MYMODEL':
    return MyModel(**kwargs)
```

3. **使用新模型**:
```bash
python main.py --model MyModel
```

### 添加新的数据划分策略

1. **在 `data/dataset.py` 定义函数**:
```python
def split_data_mystrategy(data, labels, ...):
    # 实现划分逻辑
    return train_data, val_data, test_data, ...
```

2. **在 `split_data()` 注册**:
```python
elif strategy == 'mystrategy':
    return split_data_mystrategy(...)
```

### 代码风格

- 遵循 PEP 8
- 使用类型提示
- 添加 docstring
- 保持函数简短 (<50行)
- 使用有意义的变量名

---

## 相关文档

### 核心文档
- [DATA_FUSION_USAGE.md](DATA_FUSION_USAGE.md) - 数据融合使用指南 ⭐ NEW
- [TCN_ARCHITECTURE.md](TCN_ARCHITECTURE.md) - TCN架构详解 ⭐ NEW
- [TCN_GUIDE.md](TCN_GUIDE.md) - TCN使用指南 ⭐ NEW
- [TCN_VS_OTHERS.md](TCN_VS_OTHERS.md) - TCN性能对比 ⭐ NEW
- [WINDOW_MODE_GUIDE.md](WINDOW_MODE_GUIDE.md) - Window模式训练指南
- [QUICKSTART_WINDOW.md](QUICKSTART_WINDOW.md) - Window模式快速开始

### 进阶文档
- [NO_VAL_MODE_GUIDE.md](NO_VAL_MODE_GUIDE.md) - 无验证集模式指南
- [DATASET_SELECTION_GUIDE.md](DATASET_SELECTION_GUIDE.md) - 数据集选择指南
- [TRANSFORMER_OPTIMIZATION.md](TRANSFORMER_OPTIMIZATION.md) - Transformer优化
- [OVERFITTING_SOLUTIONS.md](OVERFITTING_SOLUTIONS.md) - 过拟合解决方案

### 外部资源
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [TCN Paper](https://arxiv.org/abs/1803.01271) - An Empirical Evaluation of Generic Convolutional and Recurrent Networks

---

## 更新日志

### v2.0 (2025-12-02) ⭐ **最新版本**
- ✅ 添加TCN模型（4层，感受野31步）
- ✅ 实现数据融合功能（辅助传感器）
- ✅ 支持自定义num_layers和weight_decay
- ✅ 优化代码结构和文档
- ✅ 新增多个指南文档

### v1.3 (2025-11-28)
- ✅ 添加每个epoch在测试集上的评估
- ✅ 实时监控过拟合
- ✅ 优化W&B记录逻辑

### v1.2 (2025-11-27)
- ✅ 实现无验证集训练模式
- ✅ 添加灵活的数据集选择
- ✅ 支持自定义数据划分比例
- ✅ 优化Transformer模型

### v1.1 (2025-11-26)
- ✅ 修复数据归一化泄漏问题
- ✅ 添加4种数据划分策略
- ✅ 实现计时机制

### v1.0 (2025-11-25)
- ✅ 初始版本

---

## 许可证

MIT License

---

## 联系方式

- **项目负责人**: Tao
- **团队**: MSRA Glucose Sensing Team
- **W&B项目**: https://wandb.ai/glucose_msra/sensing

---

**最后更新**: 2025-12-02

**版本**: v2.0
