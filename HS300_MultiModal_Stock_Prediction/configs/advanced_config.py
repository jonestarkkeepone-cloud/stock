"""
高级模型配置文件
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = '../Finmultime'  # 数据在上一级目录
    market: str = 'HS300'
    stocks: List[str] = None  # None=自动加载所有790个完整四模态股票
    start_date: str = '2019-01-01'
    end_date: str = '2024-12-31'
    seq_length: int = 96
    pred_horizon: int = 24
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    
    # K线图特征配置
    use_cnn_features: bool = True
    cnn_feature_dim: int = 64
    image_size: int = 224


@dataclass
class AdvancedModelConfig:
    """高级模型配置"""
    # 模态维度
    time_series_dim: int = 6
    text_dim: int = 128  # BERT输出维度（如果不使用BERT会自动降级）
    image_dim: int = 64
    table_dim: int = 6
    
    # 融合配置
    fusion_embed_dim: int = 64
    fusion_num_heads: int = 4
    fusion_num_layers: int = 2
    fusion_output_dim: int = 256
    
    # TKAN配置
    tkan_hidden_size: int = 128
    tkan_num_layers: int = 2
    tkan_scales: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    # 特征提取器
    use_bert: bool = False  # 是否使用BERT（需要transformers库）
    use_pretrained_image: bool = False  # 是否使用预训练图像模型（需要timm库）
    
    # 不确定性估计
    estimate_uncertainty: bool = True
    
    # 正则化
    dropout: float = 0.2


@dataclass
class AdvancedTrainConfig:
    """高级训练配置"""
    # 基础训练参数
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 15
    
    # 学习率调度
    use_scheduler: bool = True
    scheduler_type: str = 'ReduceLROnPlateau'  # 'ReduceLROnPlateau' 或 'CosineAnnealing'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # 损失函数
    loss_type: str = 'mse'  # 'mse', 'mae', 'huber', 'gaussian_nll'
    use_uncertainty_loss: bool = True  # 是否使用不确定性损失
    
    # 梯度裁剪
    gradient_clip: float = 1.0
    
    # 设备和随机种子
    device: str = 'cuda'
    seed: int = 42
    
    # 保存路径
    checkpoint_dir: str = './checkpoints_advanced'
    log_dir: str = './logs_advanced'
    result_dir: str = './results_advanced'
    
    # 评估
    eval_interval: int = 1  # 每多少个epoch评估一次
    save_best_only: bool = True
    
    # 金融指标
    use_financial_metrics: bool = True
    risk_free_rate: float = 0.02


@dataclass
class AdvancedConfig:
    """主配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: AdvancedModelConfig = field(default_factory=AdvancedModelConfig)
    train: AdvancedTrainConfig = field(default_factory=AdvancedTrainConfig)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'train': self.train.__dict__
        }
    
    def print_config(self):
        """打印配置"""
        print("\n" + "=" * 70)
        print("📋 ADVANCED MODEL CONFIGURATION")
        print("=" * 70)
        
        print("\n📊 Data Configuration:")
        print(f"  Market: {self.data.market}")
        print(f"  Date Range: {self.data.start_date} to {self.data.end_date}")
        print(f"  Sequence Length: {self.data.seq_length}")
        print(f"  Prediction Horizon: {self.data.pred_horizon}")
        print(f"  Stocks: {'All (790)' if self.data.stocks is None else len(self.data.stocks)}")
        
        print("\n🧠 Model Configuration:")
        print(f"  Text Encoder: {'BERT' if self.model.use_bert else 'Simple'}")
        print(f"  Image Encoder: {'Pretrained' if self.model.use_pretrained_image else 'Custom CNN'}")
        print(f"  Fusion Dim: {self.model.fusion_output_dim}")
        print(f"  TKAN Hidden: {self.model.tkan_hidden_size}")
        print(f"  TKAN Scales: {self.model.tkan_scales}")
        print(f"  Uncertainty: {self.model.estimate_uncertainty}")
        
        print("\n🎯 Training Configuration:")
        print(f"  Batch Size: {self.train.batch_size}")
        print(f"  Epochs: {self.train.epochs}")
        print(f"  Learning Rate: {self.train.learning_rate}")
        print(f"  Loss Type: {self.train.loss_type}")
        print(f"  Use Financial Metrics: {self.train.use_financial_metrics}")
        print(f"  Device: {self.train.device}")
        
        print("=" * 70 + "\n")


# 预定义配置

def get_basic_config() -> AdvancedConfig:
    """基础配置（不使用BERT和预训练模型）"""
    config = AdvancedConfig()
    config.model.use_bert = False
    config.model.use_pretrained_image = False
    config.model.estimate_uncertainty = False
    return config


def get_advanced_config() -> AdvancedConfig:
    """高级配置（使用所有高级特性）"""
    config = AdvancedConfig()
    config.model.use_bert = True
    config.model.use_pretrained_image = True
    config.model.estimate_uncertainty = True
    config.train.use_uncertainty_loss = True
    config.train.use_financial_metrics = True
    return config


def get_fast_test_config() -> AdvancedConfig:
    """快速测试配置"""
    config = AdvancedConfig()
    config.data.stocks = ['600000', '600004', '600006']  # 只用3个股票
    config.train.epochs = 3
    config.train.batch_size = 16
    config.model.use_bert = False
    config.model.use_pretrained_image = False
    return config


def get_production_config() -> AdvancedConfig:
    """生产环境配置"""
    config = AdvancedConfig()
    config.data.stocks = None  # 使用所有股票
    config.train.epochs = 200
    config.train.batch_size = 64
    config.train.patience = 20
    config.model.use_bert = True
    config.model.use_pretrained_image = True
    config.model.estimate_uncertainty = True
    return config


if __name__ == "__main__":
    # 测试配置
    print("Testing Configurations...")
    
    # 基础配置
    basic = get_basic_config()
    basic.print_config()
    
    # 高级配置
    advanced = get_advanced_config()
    advanced.print_config()
    
    # 快速测试配置
    fast = get_fast_test_config()
    fast.print_config()
    
    print("✅ All configurations tested!")

