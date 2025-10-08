# 📚 API 参考文档

## 🎯 核心API概览

### 主要模块

- [数据处理](#数据处理) - 数据加载和预处理
- [模型定义](#模型定义) - 神经网络模型
- [训练工具](#训练工具) - 训练和评估
- [配置管理](#配置管理) - 参数配置
- [工具函数](#工具函数) - 辅助功能

---

## 数据处理

### FinMultiTimeDataset

多模态金融时序数据集类。

```python
from data.dataset import FinMultiTimeDataset

dataset = FinMultiTimeDataset(
    data_dir: str,              # 数据目录路径
    market: str,                # 市场名称（如"HS300"）
    stocks: List[str] = None,   # 股票列表，None表示加载所有
    start_date: str,            # 开始日期 "YYYY-MM-DD"
    end_date: str,              # 结束日期 "YYYY-MM-DD"
    seq_length: int = 30,       # 历史序列长度
    pred_horizon: int = 1,      # 预测时间跨度
    use_cnn_features: bool = True,  # 是否使用CNN提取图像特征
    cnn_feature_dim: int = 64   # CNN特征维度
)
```

**方法**:
- `__len__()` → int: 返回数据集大小
- `__getitem__(idx)` → dict: 获取单个样本
- `get_scaler(key)` → StandardScaler: 获取标准化器

**返回格式**:
```python
{
    'x': torch.Tensor,      # 输入特征 (seq_length, feature_dim)
    'y': torch.Tensor,      # 目标值 (pred_horizon,)
    'stock': str,           # 股票代码
    'date': str,            # 日期
    'scaler_key': str       # 标准化器键名
}
```

### create_dataloaders

创建训练、验证、测试数据加载器。

```python
from data.dataloader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    dataset: FinMultiTimeDataset,   # 数据集
    batch_size: int = 32,           # 批量大小
    train_ratio: float = 0.7,       # 训练集比例
    val_ratio: float = 0.2,         # 验证集比例
    num_workers: int = 4,           # 工作进程数
    pin_memory: bool = True         # 是否固定内存
)
```

---

## 模型定义

### MultimodalTKANModel

基础多模态TKAN模型。

```python
from models.tkan_model import MultimodalTKANModel

model = MultimodalTKANModel(
    input_size: int,        # 输入特征维度
    hidden_size: int,       # 隐藏层维度
    output_size: int,       # 输出维度
    num_layers: int = 2,    # TKAN层数
    dropout: float = 0.2    # Dropout比例
)
```

**方法**:
- `forward(x)` → torch.Tensor: 前向传播
- `get_attention_weights()` → torch.Tensor: 获取注意力权重

### AdvancedMultimodalStockPredictor

高级多模态股票预测模型。

```python
from models.advanced_model import AdvancedMultimodalStockPredictor

model = AdvancedMultimodalStockPredictor(
    time_series_dim: int = 6,       # 时序特征维度
    text_dim: int = 128,            # 文本特征维度
    image_dim: int = 64,            # 图像特征维度
    table_dim: int = 6,             # 财务特征维度
    fusion_embed_dim: int = 64,     # 融合嵌入维度
    fusion_num_heads: int = 4,      # 注意力头数
    tkan_hidden_size: int = 128,    # TKAN隐藏维度
    use_bert: bool = False,         # 是否使用BERT
    use_pretrained_image: bool = False,  # 是否使用预训练图像模型
    estimate_uncertainty: bool = True    # 是否估计不确定性
)
```

**返回格式**:
```python
{
    'mean': torch.Tensor,           # 预测均值
    'variance': torch.Tensor,       # 预测方差（如果启用）
    'confidence': torch.Tensor,     # 置信区间（如果启用）
    'attention_weights': dict       # 注意力权重
}
```

### 文本编码器

```python
from models.text_encoder import create_text_encoder

text_encoder = create_text_encoder(
    use_bert: bool = True,          # 是否使用BERT
    output_dim: int = 128,          # 输出维度
    model_name: str = 'bert-base-chinese',  # 预训练模型名称
    max_length: int = 128,          # 最大文本长度
    freeze_bert: bool = False       # 是否冻结BERT参数
)
```

### 图像编码器

```python
from models.image_encoder_advanced import create_image_encoder

image_encoder = create_image_encoder(
    feature_dim: int = 64,          # 输出特征维度
    use_pretrained: bool = True,    # 是否使用预训练模型
    pretrained_model: str = 'efficientnet_b0',  # 预训练模型名称
    use_ensemble: bool = False      # 是否使用集成模型
)
```

---

## 训练工具

### 训练函数

```python
# 基础训练
from train import train_model

train_model(
    config: Config,                 # 配置对象
    model: nn.Module = None,        # 模型（可选）
    dataset: Dataset = None         # 数据集（可选）
)

# 高级训练
from train_advanced import train_advanced_model

model = train_advanced_model(
    config: AdvancedConfig,         # 高级配置对象
    resume_from: str = None         # 恢复训练的检查点路径
)
```

### 评估函数

```python
from evaluate import evaluate_model

results = evaluate_model(
    model: nn.Module,               # 训练好的模型
    test_loader: DataLoader,        # 测试数据加载器
    device: str = 'cuda',           # 设备
    save_predictions: bool = True   # 是否保存预测结果
)
```

**返回格式**:
```python
{
    'mse': float,                   # 均方误差
    'mae': float,                   # 平均绝对误差
    'rmse': float,                  # 均方根误差
    'r2': float,                    # 决定系数
    'direction_accuracy': float,    # 方向准确率
    'sharpe_ratio': float,          # 夏普比率
    'max_drawdown': float,          # 最大回撤
    'predictions': np.ndarray,      # 预测值
    'targets': np.ndarray           # 真实值
}
```

---

## 配置管理

### 基础配置

```python
from configs.config import Config

config = Config()

# 数据配置
config.data.data_dir = "../Finmultime"
config.data.market = "HS300"
config.data.stocks = None  # 所有股票
config.data.start_date = "2019-01-01"
config.data.end_date = "2023-12-31"
config.data.seq_length = 30
config.data.pred_horizon = 1

# 模型配置
config.model.input_size = 77
config.model.hidden_size = 128
config.model.output_size = 1
config.model.num_layers = 2
config.model.dropout = 0.2

# 训练配置
config.train.batch_size = 32
config.train.epochs = 100
config.train.learning_rate = 0.001
config.train.device = 'cuda'
config.train.patience = 10
```

### 高级配置

```python
from configs.advanced_config import get_advanced_config, get_fast_test_config

# 完整高级配置
config = get_advanced_config()

# 快速测试配置
config = get_fast_test_config()

# 自定义配置
config = get_advanced_config()
config.model.use_bert = True
config.model.use_pretrained_image = True
config.train.mixed_precision = True
```

---

## 工具函数

### 评估指标

```python
from utils.metrics import calculate_comprehensive_metrics
from utils.financial_metrics import calculate_financial_metrics

# 基础指标
basic_metrics = calculate_comprehensive_metrics(
    y_true: torch.Tensor,           # 真实值
    y_pred: torch.Tensor            # 预测值
)

# 金融指标
financial_metrics = calculate_financial_metrics(
    returns_true: np.ndarray,       # 真实收益率
    returns_pred: np.ndarray,       # 预测收益率
    prices_true: np.ndarray = None, # 真实价格（可选）
    prices_pred: np.ndarray = None  # 预测价格（可选）
)
```

### 可视化工具

```python
from utils.visualization import plot_training_history, plot_predictions

# 绘制训练历史
plot_training_history(
    train_losses: List[float],      # 训练损失
    val_losses: List[float],        # 验证损失
    save_path: str = None           # 保存路径
)

# 绘制预测结果
plot_predictions(
    y_true: np.ndarray,             # 真实值
    y_pred: np.ndarray,             # 预测值
    dates: List[str] = None,        # 日期列表
    stock_name: str = None,         # 股票名称
    save_path: str = None           # 保存路径
)
```

### 数据预处理

```python
from utils.preprocessing import MultimodalPreprocessor

preprocessor = MultimodalPreprocessor()

# 时序数据标准化
ts_normalized = preprocessor.fit_transform_time_series(
    data: np.ndarray,               # 时序数据
    key: str                        # 标准化器键名
)

# 财务数据标准化
fund_normalized = preprocessor.fit_transform_fundamentals(
    data: np.ndarray,               # 财务数据
    key: str                        # 标准化器键名
)

# 反标准化
original_data = preprocessor.inverse_transform(
    data: np.ndarray,               # 标准化后的数据
    key: str                        # 标准化器键名
)
```

---

## 🔧 使用示例

### 完整训练流程

```python
import torch
from configs.config import Config
from data.dataset import FinMultiTimeDataset
from data.dataloader import create_dataloaders
from models.tkan_model import MultimodalTKANModel
from utils.metrics import calculate_comprehensive_metrics

# 1. 加载配置
config = Config()

# 2. 创建数据集
dataset = FinMultiTimeDataset(
    data_dir=config.data.data_dir,
    market=config.data.market,
    stocks=config.data.stocks,
    start_date=config.data.start_date,
    end_date=config.data.end_date,
    seq_length=config.data.seq_length,
    pred_horizon=config.data.pred_horizon
)

# 3. 创建数据加载器
train_loader, val_loader, test_loader = create_dataloaders(
    dataset, 
    batch_size=config.train.batch_size
)

# 4. 创建模型
model = MultimodalTKANModel(
    input_size=config.model.input_size,
    hidden_size=config.model.hidden_size,
    output_size=config.model.output_size,
    num_layers=config.model.num_layers
)

# 5. 训练模型
device = torch.device(config.train.device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
criterion = torch.nn.MSELoss()

for epoch in range(config.train.epochs):
    # 训练循环...
    pass

# 6. 评估模型
model.eval()
predictions, targets = [], []
with torch.no_grad():
    for batch in test_loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        pred = model(x)
        predictions.append(pred.cpu())
        targets.append(y.cpu())

predictions = torch.cat(predictions)
targets = torch.cat(targets)

# 7. 计算指标
metrics = calculate_comprehensive_metrics(targets, predictions)
print(f"R²: {metrics['r2']:.4f}")
print(f"MSE: {metrics['mse']:.4f}")
```

### 高级模型使用

```python
from configs.advanced_config import get_advanced_config
from models.advanced_model import AdvancedMultimodalStockPredictor

# 加载高级配置
config = get_advanced_config()

# 创建高级模型
model = AdvancedMultimodalStockPredictor(
    time_series_dim=config.model.time_series_dim,
    text_dim=config.model.text_dim,
    image_dim=config.model.image_dim,
    table_dim=config.model.table_dim,
    use_bert=config.model.use_bert,
    use_pretrained_image=config.model.use_pretrained_image,
    estimate_uncertainty=config.model.estimate_uncertainty
)

# 前向传播
output = model(batch_data)
mean_pred = output['mean']
if config.model.estimate_uncertainty:
    variance_pred = output['variance']
    confidence_interval = output['confidence']
```

---

## 📝 注意事项

### 数据格式要求

- **时序数据**: CSV格式，包含Date, Open, High, Low, Close, Volume列
- **图像数据**: PNG格式，灰度或RGB K线图
- **文本数据**: JSONL格式，每行包含日期和新闻文本
- **财务数据**: JSONL格式，包含基本面指标

### 内存管理

- 大批量训练时注意GPU内存限制
- 使用`pin_memory=False`减少内存使用
- 考虑使用梯度累积模拟大批量

### 模型保存

```python
# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'epoch': epoch
}, 'model_checkpoint.pt')

# 加载模型
checkpoint = torch.load('model_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

**API版本**: 2.0.0  
**最后更新**: 2025-10-08  
**兼容性**: PyTorch 1.8+, Python 3.8+
