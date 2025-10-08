"""
高级多模态股票预测模型
整合所有改进：BERT文本、注意力融合、多尺度TKAN、不确定性估计
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from models.text_encoder import create_text_encoder, SimpleSentimentEncoder
from models.image_encoder_advanced import create_image_encoder
from models.fusion import AdvancedMultimodalFusion, AdaptiveModalityFusion
from models.multiscale_tkan import MultiScaleMultimodalTKAN
from models.uncertainty import UncertaintyHead


class AdvancedMultimodalStockPredictor(nn.Module):
    """
    高级多模态股票预测器
    
    架构流程:
    1. 各模态独立编码（时序、文本、图像、财务）
    2. 高级融合（注意力机制 + 门控融合）
    3. 多尺度时序建模（TKAN）
    4. 不确定性估计（均值 + 方差）
    """
    
    def __init__(
        self,
        # 模态维度配置
        time_series_dim: int = 6,
        text_dim: int = 128,  # BERT输出维度
        image_dim: int = 64,
        table_dim: int = 6,
        
        # 融合配置
        fusion_embed_dim: int = 64,
        fusion_num_heads: int = 4,
        fusion_output_dim: int = 256,
        
        # TKAN配置
        tkan_hidden_size: int = 128,
        tkan_num_layers: int = 2,
        tkan_scales: list = None,
        
        # 输出配置
        output_size: int = 24,
        
        # 特征提取器配置
        use_bert: bool = False,  # 是否使用BERT（需要transformers库）
        use_pretrained_image: bool = False,  # 是否使用预训练图像模型
        
        # 不确定性估计
        estimate_uncertainty: bool = True,
        
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.estimate_uncertainty = estimate_uncertainty
        
        # 1. 文本编码器
        if use_bert:
            try:
                self.text_encoder = create_text_encoder(
                    use_bert=True,
                    output_dim=text_dim
                )
                print("✅ Using BERT text encoder")
            except:
                print("⚠️  BERT not available, using simple encoder")
                self.text_encoder = SimpleSentimentEncoder(output_dim=text_dim)
        else:
            self.text_encoder = SimpleSentimentEncoder(output_dim=text_dim)
        
        # 2. 图像编码器
        self.image_encoder = create_image_encoder(
            feature_dim=image_dim,
            use_pretrained=use_pretrained_image
        )
        
        # 3. 时序和财务特征投影
        self.time_series_projection = nn.Sequential(
            nn.Linear(time_series_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, time_series_dim)
        )
        
        self.table_projection = nn.Sequential(
            nn.Linear(table_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, table_dim)
        )
        
        # 4. 高级多模态融合
        modality_dims = {
            'time_series': time_series_dim,
            'text': text_dim,
            'image': image_dim,
            'table': table_dim
        }
        
        self.fusion = AdvancedMultimodalFusion(
            modality_dims=modality_dims,
            embed_dim=fusion_embed_dim,
            num_heads=fusion_num_heads,
            num_fusion_layers=2,
            output_dim=fusion_output_dim,
            dropout=dropout
        )
        
        # 5. 多尺度TKAN
        if tkan_scales is None:
            tkan_scales = [1, 3, 5]
        
        self.multiscale_tkan = MultiScaleMultimodalTKAN(
            input_size=fusion_output_dim,
            hidden_size=tkan_hidden_size,
            output_size=output_size,
            num_layers=tkan_num_layers,
            scales=tkan_scales,
            dropout=dropout
        )
        
        # 6. 不确定性估计（可选）
        if estimate_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                input_dim=fusion_output_dim,  # 使用融合后的维度
                output_dim=output_size
            )
    
    def extract_modality_features(
        self,
        time_series: torch.Tensor,
        text: torch.Tensor,
        image: torch.Tensor,
        table: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        提取各模态特征
        
        Args:
            time_series: (batch, seq_len, 6)
            text: (batch, seq_len, 1) 情感得分
            image: (batch, 1, H, W) K线图
            table: (batch, seq_len, 6) 财务数据
        
        Returns:
            features: 各模态特征字典
        """
        batch_size, seq_len, _ = time_series.shape
        
        # 1. 时序特征（保持序列）
        ts_features = self.time_series_projection(time_series)  # (batch, seq_len, 6)
        
        # 2. 文本特征（从情感得分扩展）
        text_features = self.text_encoder(text)  # (batch, seq_len, text_dim)
        
        # 3. 图像特征（每个样本一张图）
        img_features = self.image_encoder(image)  # (batch, image_dim)
        # 扩展到序列
        img_features = img_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 4. 财务特征
        table_features = self.table_projection(table)  # (batch, seq_len, 6)
        
        return {
            'time_series': ts_features,
            'text': text_features,
            'image': img_features,
            'table': table_features
        }
    
    def forward(
        self,
        time_series: torch.Tensor,
        text: torch.Tensor,
        image: torch.Tensor,
        table: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            time_series: (batch, seq_len, 6)
            text: (batch, seq_len, 1)
            image: (batch, 1, H, W)
            table: (batch, seq_len, 6)
            return_uncertainty: 是否返回不确定性
        
        Returns:
            predictions: (batch, output_size) 预测值
            uncertainty: (batch, output_size) 不确定性（可选）
        """
        batch_size, seq_len, _ = time_series.shape
        
        # 1. 提取各模态特征
        modality_features = self.extract_modality_features(
            time_series, text, image, table
        )
        
        # 2. 逐时间步融合（简化版：只融合最后时刻）
        # 取每个模态的最后时刻特征
        last_step_features = {
            k: v[:, -1, :] for k, v in modality_features.items()
        }

        # 融合
        fused_features = self.fusion(last_step_features)  # (batch, fusion_output_dim)

        # 扩展到序列 - 使用contiguous避免内存问题
        fused_sequence = fused_features.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous()
        
        # 3. 多尺度TKAN
        predictions = self.multiscale_tkan(fused_sequence)  # (batch, output_size)

        # 4. 不确定性估计（可选）
        if return_uncertainty and self.estimate_uncertainty:
            # 使用融合后的最后时刻特征估计不确定性
            # fused_features: (batch, fusion_output_dim)
            mean, var = self.uncertainty_head(fused_features)
            return mean, var

        return predictions, None
    
    def predict_with_confidence(
        self,
        time_series: torch.Tensor,
        text: torch.Tensor,
        image: torch.Tensor,
        table: torch.Tensor,
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测并返回置信区间
        
        Returns:
            mean: 预测均值
            lower: 下界
            upper: 上界
        """
        if not self.estimate_uncertainty:
            predictions, _ = self.forward(time_series, text, image, table)
            return predictions, predictions, predictions
        
        mean, var = self.forward(
            time_series, text, image, table, 
            return_uncertainty=True
        )
        
        std = torch.sqrt(var)
        
        # 计算置信区间（假设正态分布）
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return mean, lower, upper


def create_advanced_model(config: dict) -> AdvancedMultimodalStockPredictor:
    """
    工厂函数：根据配置创建高级模型
    
    Args:
        config: 配置字典
    
    Returns:
        model: 高级多模态模型
    """
    model = AdvancedMultimodalStockPredictor(
        time_series_dim=config.get('time_series_dim', 6),
        text_dim=config.get('text_dim', 128),
        image_dim=config.get('image_dim', 64),
        table_dim=config.get('table_dim', 6),
        fusion_embed_dim=config.get('fusion_embed_dim', 64),
        fusion_output_dim=config.get('fusion_output_dim', 256),
        tkan_hidden_size=config.get('tkan_hidden_size', 128),
        tkan_num_layers=config.get('tkan_num_layers', 2),
        output_size=config.get('output_size', 24),
        use_bert=config.get('use_bert', False),
        use_pretrained_image=config.get('use_pretrained_image', False),
        estimate_uncertainty=config.get('estimate_uncertainty', True),
        dropout=config.get('dropout', 0.2)
    )
    
    # 统计参数
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"🚀 Advanced Multimodal Stock Predictor Created")
    print(f"{'='*70}")
    print(f"📊 Model Configuration:")
    print(f"  - Time Series Dim: {config.get('time_series_dim', 6)}")
    print(f"  - Text Dim: {config.get('text_dim', 128)}")
    print(f"  - Image Dim: {config.get('image_dim', 64)}")
    print(f"  - Table Dim: {config.get('table_dim', 6)}")
    print(f"  - Fusion Output: {config.get('fusion_output_dim', 256)}")
    print(f"  - TKAN Hidden: {config.get('tkan_hidden_size', 128)}")
    print(f"  - Output Size: {config.get('output_size', 24)}")
    print(f"\n📈 Model Statistics:")
    print(f"  - Total Parameters: {num_params:,}")
    print(f"  - Uncertainty Estimation: {config.get('estimate_uncertainty', True)}")
    print(f"{'='*70}\n")
    
    return model


if __name__ == "__main__":
    # 测试代码
    print("Testing Advanced Multimodal Model...")
    
    # 配置
    config = {
        'time_series_dim': 6,
        'text_dim': 128,
        'image_dim': 64,
        'table_dim': 6,
        'fusion_output_dim': 256,
        'tkan_hidden_size': 128,
        'output_size': 24,
        'use_bert': False,
        'use_pretrained_image': False,
        'estimate_uncertainty': True
    }
    
    # 创建模型
    model = create_advanced_model(config)
    
    # 测试数据
    batch_size = 4
    seq_len = 60
    
    time_series = torch.randn(batch_size, seq_len, 6)
    text = torch.randn(batch_size, seq_len, 1)
    image = torch.randn(batch_size, 1, 224, 224)
    table = torch.randn(batch_size, seq_len, 6)
    
    # 前向传播
    predictions, _ = model(time_series, text, image, table)
    print(f"✅ Predictions shape: {predictions.shape}")
    
    # 带不确定性的预测
    mean, var = model(time_series, text, image, table, return_uncertainty=True)
    print(f"✅ Mean shape: {mean.shape}, Var shape: {var.shape}")
    
    print("✅ All advanced model tests passed!")

