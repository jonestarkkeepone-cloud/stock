import torch
import torch.nn as nn

class HybridKLineEncoder(nn.Module):
    """
    混合架构: CNN捕获局部特征 + Self-Attention捕获全局依赖
    专门为K线图设计
    """
    def __init__(self, feature_dim=64):
        super().__init__()
        
        # CNN主干网络
        self.conv_backbone = nn.Sequential(
            # Block 1: 捕获单根K线特征
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112
            
            # Block 2: 捕获短期趋势（几根K线）
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            
            # Block 3: 捕获形态（头肩顶、双底等）
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28
        )
        
        # Self-Attention层（捕获全局依赖）
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 输出映射
        self.output_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, feature_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, H, W) 灰度K线图
        Returns:
            features: (batch, feature_dim) 特征向量
        """
        # CNN特征提取
        x = self.conv_backbone(x)  # (B, 256, 28, 28)
        
        B, C, H, W = x.shape
        
        # 重塑为序列用于Attention
        x_seq = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, 256)
        
        # Self-Attention
        x_attn, _ = self.attention(x_seq, x_seq, x_seq)
        
        # 重塑回空间维度
        x_attn = x_attn.permute(0, 2, 1).view(B, C, H, W)
        
        # 残差连接
        x = x + x_attn
        
        # 输出特征
        features = self.output_projection(x)
        
        return features