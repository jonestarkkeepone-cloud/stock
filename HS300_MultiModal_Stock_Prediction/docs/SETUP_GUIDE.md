# 🛠️ HS300 多模态股票预测系统 - 安装配置指南

## 📋 目录

1. [快速安装](#快速安装)
2. [详细配置](#详细配置)
3. [数据准备](#数据准备)
4. [配置选项](#配置选项)
5. [故障排除](#故障排除)
6. [性能优化](#性能优化)

---

## 快速安装

### ⚡ 一键安装脚本

```bash
# 1. 创建环境并安装依赖
conda create -n hs300 python=3.9 -y
conda activate hs300
pip install -r requirements.txt

# 2. 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"

# 3. 快速测试
python extract_stocks.py
python train.py --mode train --epochs 1 --stocks 1
```

### 📋 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **操作系统** | Windows 10, Linux, macOS | Ubuntu 20.04+, Windows 11 |
| **Python** | 3.8+ | 3.9-3.11 |
| **内存** | 8GB | 32GB+ |
| **GPU** | 可选 | NVIDIA RTX 3080+ |
| **存储** | 20GB | 100GB+ SSD |
| **CUDA** | 可选 | 11.8+ |

---

## 详细配置

### 1. 环境准备

#### 方式一：Conda环境（推荐）

```bash
# 创建新环境
conda create -n hs300 python=3.9 -y
conda activate hs300

# 安装PyTorch（根据您的CUDA版本选择）
# CPU版本
conda install pytorch torchvision cpuonly -c pytorch

# GPU版本（CUDA 11.8）
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

#### 方式二：虚拟环境

```bash
# 创建虚拟环境
python -m venv hs300_env

# 激活环境
# Windows
hs300_env\Scripts\activate
# Linux/Mac
source hs300_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 依赖安装

#### 基础安装（核心功能）

```bash
pip install -r requirements.txt
```

#### 高级功能安装（可选）

```bash
# BERT文本编码器
pip install transformers tokenizers

# 预训练图像模型
pip install timm

# 开发工具
pip install tensorboard wandb pytest black flake8
```

### 3. 验证安装

```bash
# 检查核心依赖
python -c "
import torch
import numpy as np
import pandas as pd
import sklearn
print('✅ 核心依赖安装成功')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
"

# 检查高级功能（可选）
python -c "
try:
    import transformers
    print('✅ Transformers可用')
except ImportError:
    print('⚠️ Transformers未安装（高级文本功能不可用）')

try:
    import timm
    print('✅ TIMM可用')
except ImportError:
    print('⚠️ TIMM未安装（高级图像功能不可用）')
"
```

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 数据准备

### 数据结构

项目需要 `Finmultime` 数据目录，结构如下：

```
Finmultime/
├── time_series/
│   └── HS300_time_series/
│       └── HS300_time_series/
│           ├── 600000.SS.csv
│           ├── 000001.SZ.csv
│           └── ...
├── image/
│   └── HS300_image/
│       └── HS300_image/
│           ├── 600000.SS_2019-01-02.png
│           └── ...
├── text/
│   └── hs300news_summary/
│       └── hs300news_summary/
│           ├── 600000.SS.jsonl
│           └── ...
└── table/
    └── hs300_tabular/
        └── hs300_tabular/
            ├── 600000.SS/
            │   └── income.jsonl
            └── ...
```

### 数据放置

**选项1**: 将 `Finmultime` 目录放在项目根目录的上一级

```
parent_directory/
├── Finmultime/          # 数据目录
└── HS300_MultiModal_Stock_Prediction/  # 项目目录
```

**选项2**: 修改配置文件中的路径

编辑 `configs/config.py`:

```python
@dataclass
class DataConfig:
    data_dir: str = 'E:/path/to/your/Finmultime'  # 修改为实际路径
    ...
```

### 提取完整四模态股票

运行提取脚本：

```bash
python extract_stocks.py
```

这将生成 `hs300_complete_stocks.json`，包含790个完整四模态股票的信息。

---

## 配置说明

### configs/config.py

#### DataConfig - 数据配置

```python
@dataclass
class DataConfig:
    # 数据目录
    data_dir: str = '../Finmultime'
    
    # 市场选择
    market: str = 'HS300'  # 'HS300' 或 'SP500'
    
    # 股票列表
    stocks: List[str] = None  # None=自动加载所有790个股票
    # 或指定特定股票: ['600000', '600004', '600006']
    
    # 日期范围
    start_date: str = '2019-01-01'
    end_date: str = '2025-12-31'
    
    # 序列参数
    seq_length: int = 60        # 输入序列长度（天）
    pred_horizon: int = 24      # 预测步长（天）
    
    # 特征配置
    use_cnn_features: bool = True   # 是否使用CNN提取图像特征
    cnn_feature_dim: int = 64       # CNN特征维度
```

#### ModelConfig - 模型配置

```python
@dataclass
class ModelConfig:
    # 输入维度: 6(ts) + 1(text) + 64(image) + 6(table) = 77
    input_size: int = 77
    
    # 隐藏层维度
    hidden_size: int = 128
    
    # TKAN层数
    num_layers: int = 2
    
    # Dropout率
    dropout: float = 0.2
```

#### TrainConfig - 训练配置

```python
@dataclass
class TrainConfig:
    # 批次大小
    batch_size: int = 32
    
    # 训练轮数
    epochs: int = 100
    
    # 学习率
    learning_rate: float = 0.001
    
    # 权重衰减
    weight_decay: float = 1e-5
    
    # 早停耐心值
    patience: int = 10
    
    # 随机种子
    seed: int = 42
    
    # 保存目录
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
```

---

## 快速开始

### 1. 快速训练测试 (5分钟)

使用3个股票，训练3个epoch：

```bash
python quick_train.py --epochs 3 --test_stocks 3
```

### 2. 中等规模训练 (30分钟)

使用10个股票，训练10个epoch：

```bash
python quick_train.py --epochs 10 --test_stocks 10 --batch_size 32
```

### 3. 完整训练 (数小时)

使用所有790个股票，训练100个epoch：

```bash
python train.py --epochs 100 --batch_size 64
```

### 4. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### 5. 生成可视化

```bash
python visualize.py --mode all
```

---

## 高级配置

### 使用GPU加速

确保安装了CUDA版本的PyTorch：

```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 如果不可用，安装CUDA版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 自定义股票列表

#### 方法1: 修改配置文件

编辑 `configs/config.py`:

```python
@dataclass
class DataConfig:
    stocks: List[str] = ['600000', '600004', '600006', '600008', '600009']
```

#### 方法2: 按行业筛选

```python
import json

# 加载股票信息
with open('hs300_complete_stocks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 筛选医药生物行业
pharma_stocks = [
    stock['证券代码'].replace('.SS', '').replace('.SZ', '')
    for stock in data['stock_info']
    if stock['申万一级行业'] == '医药生物'
]

print(f"医药生物行业股票: {len(pharma_stocks)}个")
print(pharma_stocks[:10])
```

### 调整超参数

#### 学习率调度

在 `train.py` 中添加：

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5
)

# 在训练循环中
for epoch in range(epochs):
    # ... 训练代码 ...
    scheduler.step(val_loss)
```

#### 早停

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 使用
early_stopping = EarlyStopping(patience=10)
for epoch in range(epochs):
    # ... 训练代码 ...
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

---

## 故障排除

### 问题1: 数据集为空

**症状**: `数据集创建成功，总样本数: 0`

**原因**:
- 数据文件路径不正确
- 日期范围不匹配
- 文件命名格式不对

**解决方案**:

```bash
# 1. 检查数据目录
ls ../Finmultime/time_series/HS300_time_series/HS300_time_series/ | head

# 2. 检查文件格式
python -c "
import pandas as pd
df = pd.read_csv('../Finmultime/time_series/HS300_time_series/HS300_time_series/600000.SS.csv')
print('Columns:', df.columns.tolist())
print('Date range:', df['Date'].min(), 'to', df['Date'].max())
"

# 3. 验证股票代码
python -c "
from pathlib import Path
ts_dir = Path('../Finmultime/time_series/HS300_time_series/HS300_time_series')
files = list(ts_dir.glob('*.csv'))
print(f'Found {len(files)} CSV files')
print('First 5:', [f.stem for f in files[:5]])
"
```

### 问题2: 内存不足

**症状**: `RuntimeError: CUDA out of memory` 或系统内存不足

**解决方案**:

1. 减少batch_size:
```bash
python quick_train.py --batch_size 8
```

2. 减少股票数量:
```bash
python quick_train.py --test_stocks 5
```

3. 减少序列长度（修改配置）:
```python
seq_length: int = 30  # 从60减少到30
```

### 问题3: 训练速度慢

**解决方案**:

1. 使用GPU:
```bash
# 检查GPU是否被使用
python -c "import torch; print(torch.cuda.is_available())"
```

2. 增加batch_size (如果内存允许):
```bash
python train.py --batch_size 128
```

3. 使用数据并行:
```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### 问题4: 模型不收敛

**症状**: 损失不下降或波动很大

**解决方案**:

1. 降低学习率:
```python
learning_rate: float = 0.0001  # 从0.001降低
```

2. 增加训练轮数:
```bash
python train.py --epochs 200
```

3. 检查数据归一化:
```python
# 在dataset.py中确保数据已归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data)
```

### 问题5: 文件编码错误

**症状**: `UnicodeDecodeError` 或中文乱码

**解决方案**:

```python
# 读取CSV时指定编码
df = pd.read_csv(file_path, encoding='utf-8')
# 或
df = pd.read_csv(file_path, encoding='gbk')

# 读取JSON时指定编码
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
```

---

## 性能优化建议

### 1. 数据加载优化

```python
# 使用多进程加载数据
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # 增加worker数量
    pin_memory=True  # 如果使用GPU
)
```

### 2. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        
        with autocast():
            output = model(X)
            loss = criterion(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 3. 梯度累积

```python
accumulation_steps = 4

for i, (X, y) in enumerate(train_loader):
    output = model(X)
    loss = criterion(output, y) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 联系支持

如有其他问题，请：
1. 查看 [README.md](../README.md)
2. 查看 [API文档](API_REFERENCE.md)
3. 提交Issue

---

**最后更新**: 2025-10-07

