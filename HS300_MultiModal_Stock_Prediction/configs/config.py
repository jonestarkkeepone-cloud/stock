from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = './Finmultime'  # 使用HS300完整四模态数据
    market: str = 'HS300'  # 使用HS300市场数据
    stocks: List[str] = None  # None=自动加载所有790个完整四模态股票
    start_date: str = '2019-01-01'  # 2019-2025数据
    end_date: str = '2024-12-31'
    seq_length: int = 96
    pred_horizon: int = 24
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    
    # K线图特征配置
    use_cnn_features: bool = True
    cnn_feature_dim: int = 64  # 混合架构输出维度
    image_size: int = 224

@dataclass
class ModelConfig:
    """模型配置"""
    # 输入维度: 6(ts) + 1(text) + 64(image_cnn) + 6(table) = 77
    input_size: int = 77
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 10
    device: str = 'cuda'
    seed: int = 42
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'

@dataclass
class Config:
    """主配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)