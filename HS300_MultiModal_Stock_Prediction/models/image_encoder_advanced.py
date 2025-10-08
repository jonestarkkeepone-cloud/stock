"""
高级图像编码器 - 使用预训练模型和K线模式识别
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class KLinePatternAttention(nn.Module):
    """
    K线模式注意力模块
    专门识别K线图中的技术形态（头肩顶、双底等）
    """
    
    def __init__(self, in_channels: int = 256, num_patterns: int = 8):
        """
        Args:
            in_channels: 输入通道数
            num_patterns: 要识别的模式数量
        """
        super().__init__()
        
        self.num_patterns = num_patterns
        
        # 模式查询向量（可学习）
        self.pattern_queries = nn.Parameter(torch.randn(num_patterns, in_channels))
        
        # 注意力计算
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, num_patterns, 1)
        )
        
        # 模式特征提取
        self.pattern_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)
        Returns:
            pattern_features: (batch, in_channels)
        """
        B, C, H, W = x.shape
        
        # 提取模式特征
        pattern_feat = self.pattern_extractor(x)  # (B, C, H, W)
        
        # 计算注意力图
        attn_map = self.attention(x)  # (B, num_patterns, H, W)
        attn_map = F.softmax(attn_map.view(B, self.num_patterns, -1), dim=-1)
        attn_map = attn_map.view(B, self.num_patterns, H, W)
        
        # 应用注意力
        pattern_feat_flat = pattern_feat.view(B, C, -1)  # (B, C, H*W)
        attn_map_flat = attn_map.view(B, self.num_patterns, -1)  # (B, num_patterns, H*W)
        
        # 加权聚合
        weighted = torch.bmm(pattern_feat_flat, attn_map_flat.transpose(1, 2))  # (B, C, num_patterns)
        
        # 全局池化
        output = weighted.mean(dim=-1)  # (B, C)
        
        return output


class TechnicalIndicatorModule(nn.Module):
    """
    技术指标模块
    从图像特征中提取技术指标相关的信息
    """
    
    def __init__(self, in_channels: int = 256, out_features: int = 32):
        super().__init__()
        
        # 趋势检测
        self.trend_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 8)),  # 水平方向保留更多信息
            nn.Flatten(),
            nn.Linear(in_channels * 8, out_features)
        )
        
        # 波动性检测
        self.volatility_detector = nn.Sequential(
            nn.AdaptiveMaxPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(in_channels * 16, out_features)
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(out_features * 2, out_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)
        Returns:
            indicators: (batch, out_features)
        """
        trend = self.trend_detector(x)
        volatility = self.volatility_detector(x)
        
        combined = torch.cat([trend, volatility], dim=-1)
        indicators = self.fusion(combined)
        
        return indicators


class AdvancedKLineEncoder(nn.Module):
    """
    高级K线编码器
    使用预训练模型 + K线特定模块
    """
    
    def __init__(
        self, 
        feature_dim: int = 64,
        use_pretrained: bool = True,
        pretrained_model: str = 'efficientnet_b0',
        freeze_backbone: bool = False
    ):
        """
        Args:
            feature_dim: 输出特征维度
            use_pretrained: 是否使用预训练模型
            pretrained_model: 预训练模型名称
            freeze_backbone: 是否冻结主干网络
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            try:
                import timm
                
                # 加载预训练模型
                print(f"Loading pretrained model: {pretrained_model}")
                self.backbone = timm.create_model(
                    pretrained_model,
                    pretrained=True,
                    in_chans=1,  # 灰度图
                    num_classes=0,  # 移除分类头
                    global_pool=''  # 保留空间维度
                )
                
                # 获取特征维度
                with torch.no_grad():
                    dummy_input = torch.randn(1, 1, 224, 224)
                    dummy_output = self.backbone(dummy_input)
                    backbone_channels = dummy_output.shape[1]
                
                print(f"✅ Loaded {pretrained_model}, output channels: {backbone_channels}")
                
                # 是否冻结主干
                if freeze_backbone:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                    print("Backbone frozen")
                
            except Exception as e:
                print(f"⚠️  Failed to load pretrained model: {e}")
                print("Falling back to custom CNN")
                self.use_pretrained = False
                backbone_channels = 256
        
        if not self.use_pretrained:
            # 自定义CNN（备用）
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            backbone_channels = 256
        
        # K线模式识别
        self.pattern_attention = KLinePatternAttention(
            in_channels=backbone_channels,
            num_patterns=8
        )
        
        # 技术指标模块
        self.technical_indicators = TechnicalIndicatorModule(
            in_channels=backbone_channels,
            out_features=32
        )
        
        # 全局特征
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征融合
        total_features = backbone_channels + 32  # pattern + indicators
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, H, W) 灰度K线图
        Returns:
            features: (batch, feature_dim)
        """
        # 主干网络提取特征
        backbone_features = self.backbone(x)  # (B, C, H', W')
        
        # K线模式特征
        pattern_features = self.pattern_attention(backbone_features)  # (B, C)
        
        # 技术指标特征
        indicator_features = self.technical_indicators(backbone_features)  # (B, 32)
        
        # 拼接所有特征
        combined = torch.cat([pattern_features, indicator_features], dim=-1)
        
        # 最终特征
        output = self.feature_fusion(combined)
        
        return output


class EnsembleKLineEncoder(nn.Module):
    """
    集成K线编码器
    使用多个编码器的集成来提高鲁棒性
    """
    
    def __init__(self, feature_dim: int = 64, num_encoders: int = 3):
        super().__init__()
        
        self.num_encoders = num_encoders
        
        # 创建多个编码器
        self.encoders = nn.ModuleList([
            AdvancedKLineEncoder(
                feature_dim=feature_dim,
                use_pretrained=False  # 使用自定义CNN以避免重复加载
            )
            for _ in range(num_encoders)
        ])
        
        # 集成权重
        self.ensemble_weights = nn.Parameter(torch.ones(num_encoders) / num_encoders)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, H, W)
        Returns:
            features: (batch, feature_dim)
        """
        # 获取所有编码器的输出
        outputs = [encoder(x) for encoder in self.encoders]
        
        # 堆叠
        stacked = torch.stack(outputs, dim=0)  # (num_encoders, batch, feature_dim)
        
        # 加权平均
        weights = F.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        ensemble_output = (stacked * weights).sum(dim=0)
        
        return ensemble_output


def create_image_encoder(
    feature_dim: int = 64,
    use_pretrained: bool = True,
    use_ensemble: bool = False,
    **kwargs
) -> nn.Module:
    """
    工厂函数：创建图像编码器
    
    Args:
        feature_dim: 输出特征维度
        use_pretrained: 是否使用预训练模型
        use_ensemble: 是否使用集成模型
        **kwargs: 其他参数
    
    Returns:
        encoder: 图像编码器
    """
    if use_ensemble:
        encoder = EnsembleKLineEncoder(feature_dim=feature_dim)
        print(f"✅ Created ensemble K-line encoder (feature_dim={feature_dim})")
    else:
        encoder = AdvancedKLineEncoder(
            feature_dim=feature_dim,
            use_pretrained=use_pretrained,
            **kwargs
        )
        print(f"✅ Created advanced K-line encoder (feature_dim={feature_dim})")
    
    return encoder


if __name__ == "__main__":
    # 测试代码
    print("Testing Image Encoders...")
    
    # 测试高级编码器
    encoder = create_image_encoder(feature_dim=64, use_pretrained=False)
    test_image = torch.randn(2, 1, 224, 224)
    features = encoder(test_image)
    print(f"Advanced encoder output shape: {features.shape}")
    
    # 测试集成编码器
    ensemble_encoder = create_image_encoder(feature_dim=64, use_ensemble=True)
    features = ensemble_encoder(test_image)
    print(f"Ensemble encoder output shape: {features.shape}")
    
    print("✅ All image encoder tests passed!")

