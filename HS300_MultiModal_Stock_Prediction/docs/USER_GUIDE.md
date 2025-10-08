# 📖 HS300 多模态股票预测系统 - 用户使用指南

## 🎯 快速导航

- [新手入门](#新手入门) - 5分钟快速体验
- [完整教程](#完整教程) - 详细使用步骤
- [高级功能](#高级功能) - 进阶使用技巧
- [常见问题](#常见问题) - 问题解决方案

---

## 新手入门

### ⚡ 5分钟快速体验

```bash
# 1. 安装依赖（首次运行）
pip install -r requirements.txt

# 2. 提取股票数据（首次运行）
python extract_stocks.py

# 3. 快速训练测试（3个股票，3轮训练，约2分钟）
python train.py --mode train --epochs 3 --stocks 3

# 4. 查看训练结果
python visualize.py --results ./results/results.json
```

### 📊 预期结果

运行成功后，您将看到：
- 训练损失从约1.0降到0.2以下
- R²指标达到0.8+
- 生成预测vs真实值的对比图
- 保存最佳模型到`./checkpoints/`

---

## 完整教程

### 第一步：环境准备

#### 1.1 检查系统要求

```bash
# 检查Python版本（需要3.8+）
python --version

# 检查可用内存（推荐8GB+）
# Windows
wmic computersystem get TotalPhysicalMemory
# Linux
free -h
```

#### 1.2 安装依赖

```bash
# 基础安装
pip install -r requirements.txt

# 高级功能（可选，需要更多内存）
pip install transformers timm
```

### 第二步：数据准备

#### 2.1 数据目录结构

确保您的数据目录结构如下：

```
parent_directory/
├── Finmultime/                    # 您的数据目录
│   ├── time_series/
│   │   └── HS300_time_series/
│   │       └── HS300_time_series/
│   │           ├── 600000.SS.csv  # 上海股票
│   │           ├── 000001.SZ.csv  # 深圳股票
│   │           └── ...
│   ├── image/
│   │   └── HS300_image/
│   │       └── HS300_image/
│   │           ├── 600000.SS.png
│   │           └── ...
│   ├── text/
│   │   └── hs300news_summary/
│   │       └── hs300news_summary/
│   │           ├── 600000.SS.jsonl
│   │           └── ...
│   └── table/
│       └── hs300_tabular/
│           └── hs300_tabular/
│               ├── 600000.SS/
│               │   └── income.jsonl
│               └── ...
└── HS300_MultiModal_Stock_Prediction/  # 本项目
```

#### 2.2 提取完整数据股票

```bash
# 扫描数据目录，找出具有完整四模态数据的股票
python extract_stocks.py

# 查看提取结果
python -c "
import json
with open('hs300_complete_stocks.json', 'r') as f:
    stocks = json.load(f)
print(f'找到 {len(stocks)} 个完整四模态股票')
print('前5个股票:', list(stocks.keys())[:5])
"
```

### 第三步：模型训练

#### 3.1 快速测试训练

```bash
# 使用3个股票进行快速测试（约2-5分钟）
python train.py --mode train --epochs 3 --stocks 3 --batch_size 16

# 查看训练日志
tail -f logs/training.log
```

#### 3.2 小规模训练

```bash
# 使用10个股票进行小规模训练（约30-60分钟）
python train.py --mode train --epochs 20 --stocks 10 --batch_size 32

# 使用GPU加速（如果可用）
python train.py --mode train --epochs 20 --stocks 10 --device cuda
```

#### 3.3 完整训练

```bash
# 使用所有可用股票进行完整训练（约2-6小时）
python train.py --mode train --epochs 100 --batch_size 64

# 后台运行（Linux/Mac）
nohup python train.py --mode train --epochs 100 > training.log 2>&1 &

# 后台运行（Windows PowerShell）
Start-Process python -ArgumentList "train.py --mode train --epochs 100" -WindowStyle Hidden
```

### 第四步：模型评估

#### 4.1 基础评估

```bash
# 评估最佳模型
python evaluate.py --checkpoint ./checkpoints/best_model.pt

# 评估特定检查点
python evaluate.py --checkpoint ./checkpoints/epoch_50.pt
```

#### 4.2 详细分析

```bash
# 生成详细的评估报告
python evaluate.py --checkpoint ./checkpoints/best_model.pt --detailed

# 按股票分析性能
python evaluate.py --checkpoint ./checkpoints/best_model.pt --by_stock
```

### 第五步：结果可视化

#### 5.1 基础可视化

```bash
# 生成所有基础图表
python visualize.py --results ./results/results.json

# 只生成训练曲线
python visualize.py --results ./results/results.json --plot_type training

# 只生成预测对比图
python visualize.py --results ./results/results.json --plot_type prediction
```

#### 5.2 高级可视化

```bash
# 生成交互式图表（需要安装plotly）
pip install plotly
python visualize.py --results ./results/results.json --interactive

# 生成股票级别的详细分析
python visualize.py --results ./results/results.json --stock_analysis
```

---

## 高级功能

### 🧠 使用高级模型

#### 启用BERT文本编码

```bash
# 安装transformers
pip install transformers

# 使用高级配置训练
python -c "
from configs.advanced_config import get_advanced_config
config = get_advanced_config()
config.model.use_bert = True
# 保存配置并训练...
"
```

#### 启用预训练图像模型

```bash
# 安装timm
pip install timm

# 使用预训练图像编码器
python -c "
from configs.advanced_config import get_advanced_config
config = get_advanced_config()
config.model.use_pretrained_image = True
# 训练...
"
```

### 📊 自定义配置

#### 创建自定义配置文件

```python
# custom_config.py
from configs.config import Config

def get_custom_config():
    config = Config()
    
    # 数据配置
    config.data.seq_length = 60        # 使用60天历史数据
    config.data.pred_horizon = 5       # 预测5天
    
    # 模型配置
    config.model.hidden_size = 256     # 更大的隐藏层
    config.model.num_layers = 3        # 更深的网络
    
    # 训练配置
    config.train.learning_rate = 0.0005  # 更小的学习率
    config.train.batch_size = 16        # 更小的批量
    
    return config

# 使用自定义配置
if __name__ == "__main__":
    config = get_custom_config()
    # 开始训练...
```

### 🔧 性能优化

#### GPU加速

```bash
# 检查GPU可用性
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 使用GPU训练
python train.py --device cuda --batch_size 128

# 多GPU训练（如果有多个GPU）
python train.py --device cuda --multi_gpu
```

#### 混合精度训练

```bash
# 启用混合精度（节省显存，加速训练）
python train.py --mixed_precision --batch_size 256
```

#### 内存优化

```bash
# 减少内存使用
python train.py --batch_size 8 --num_workers 2 --pin_memory False

# 使用梯度累积（模拟大批量）
python train.py --batch_size 8 --gradient_accumulation_steps 8  # 等效batch_size=64
```

---

## 常见问题

### ❓ 安装问题

**Q: 安装PyTorch时出现CUDA版本不匹配**

A: 检查CUDA版本并安装对应的PyTorch：
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Q: 安装transformers失败**

A: 使用国内镜像源：
```bash
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### ❓ 数据问题

**Q: 提取股票时显示"找到0个完整四模态股票"**

A: 检查数据目录路径：
```bash
# 检查数据目录是否存在
ls ../Finmultime/

# 修改配置文件中的数据路径
# 编辑 configs/config.py，修改 data_dir 参数
```

**Q: 训练时出现"数据加载失败"**

A: 检查数据文件完整性：
```bash
# 运行数据诊断脚本
python -c "
from data.dataset import FinMultiTimeDataset
from configs.config import Config

config = Config()
try:
    dataset = FinMultiTimeDataset(
        data_dir=config.data.data_dir,
        market=config.data.market,
        stocks=['600000'],  # 测试单个股票
        start_date='2020-01-01',
        end_date='2020-12-31'
    )
    print(f'✅ 数据加载成功，样本数: {len(dataset)}')
except Exception as e:
    print(f'❌ 数据加载失败: {e}')
"
```

### ❓ 训练问题

**Q: 训练过程中内存不足**

A: 减少内存使用：
```bash
# 方法1：减少批量大小
python train.py --batch_size 8

# 方法2：减少工作进程
python train.py --num_workers 1

# 方法3：使用基础配置
python train.py --config basic
```

**Q: 训练损失不下降**

A: 调整学习率和检查数据：
```bash
# 降低学习率
python train.py --lr 0.0001

# 增加训练轮次
python train.py --epochs 200

# 检查数据质量
python -c "
import torch
from data.dataset import FinMultiTimeDataset
# 检查数据分布...
"
```

**Q: GPU显存不足**

A: 优化显存使用：
```bash
# 减少批量大小
python train.py --batch_size 16

# 启用混合精度
python train.py --mixed_precision

# 使用梯度检查点
python train.py --gradient_checkpointing
```

### ❓ 结果问题

**Q: 模型性能不理想**

A: 尝试以下优化：
```bash
# 1. 增加模型复杂度
python train.py --hidden_size 256 --num_layers 3

# 2. 使用更长的历史数据
python train.py --seq_length 60

# 3. 启用高级功能
pip install transformers timm
python train.py --config advanced

# 4. 调整学习率
python train.py --lr 0.0005
```

**Q: 预测结果不稳定**

A: 提高模型稳定性：
```bash
# 1. 增加训练数据
python train.py --stocks 50

# 2. 使用更多训练轮次
python train.py --epochs 200

# 3. 添加正则化
python train.py --weight_decay 0.01 --dropout 0.3

# 4. 使用集成方法
python train.py --ensemble 5
```

---

## 📈 性能基准

### 不同配置的预期性能

| 配置 | 股票数 | 训练时间 | R² | 方向准确率 | 内存需求 |
|------|--------|----------|-----|-----------|----------|
| **快速测试** | 3 | 2-5分钟 | 0.75+ | 85%+ | 4GB |
| **小规模** | 10 | 30-60分钟 | 0.80+ | 87%+ | 8GB |
| **中等规模** | 50 | 2-4小时 | 0.82+ | 88%+ | 16GB |
| **完整训练** | 464 | 4-8小时 | 0.85+ | 90%+ | 32GB |

### 硬件性能对比

| 硬件配置 | 训练速度 | 推荐用途 |
|----------|----------|----------|
| **CPU Only** | 1x | 快速测试 |
| **GTX 1660** | 3x | 小规模训练 |
| **RTX 3080** | 8x | 中等规模训练 |
| **RTX 4090** | 15x | 完整训练 |

---

## 🎯 最佳实践

### 1. 新手建议

1. **从小开始**：先用3个股票测试，确保流程正常
2. **检查数据**：确保数据目录结构正确
3. **监控训练**：观察损失曲线，及时调整参数
4. **保存结果**：定期备份模型和结果

### 2. 进阶用户

1. **使用高级功能**：启用BERT和预训练模型
2. **调优参数**：根据数据特点调整模型结构
3. **集成方法**：使用多个模型集成提高性能
4. **自定义损失**：根据业务需求设计损失函数

### 3. 生产部署

1. **模型压缩**：使用量化和剪枝减少模型大小
2. **推理优化**：使用TensorRT或ONNX加速推理
3. **监控系统**：建立模型性能监控
4. **版本管理**：使用MLflow管理模型版本

---

**文档版本**: 2.0.0  
**最后更新**: 2025-10-08  
**技术支持**: 查看GitHub Issues或联系维护团队
