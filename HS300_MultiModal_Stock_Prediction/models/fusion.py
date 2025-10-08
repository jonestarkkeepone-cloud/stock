"""
高级多模态融合模块
实现Cross-Modal Attention和Gated Fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class ModalityEmbedding(nn.Module):
    """
    模态嵌入层
    将不同维度的模态特征映射到统一的嵌入空间
    """
    
    def __init__(self, modality_dims: Dict[str, int], embed_dim: int = 64):
        """
        Args:
            modality_dims: 各模态的输入维度，例如 {'time_series': 6, 'text': 128, ...}
            embed_dim: 统一的嵌入维度
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.embed_dim = embed_dim
        
        # 为每个模态创建投影层
        self.projections = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.projections[modality] = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # 模态类型嵌入（可学习的位置编码）
        self.modality_type_embeddings = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, embed_dim))
            for modality in modality_dims.keys()
        })
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            modality_features: 字典，键为模态名，值为特征张量
        Returns:
            embedded_features: 嵌入后的特征字典
        """
        embedded = {}
        for modality, features in modality_features.items():
            if modality in self.projections:
                # 投影到嵌入空间
                proj = self.projections[modality](features)
                
                # 添加模态类型嵌入
                proj = proj + self.modality_type_embeddings[modality]
                
                embedded[modality] = proj
        
        return embedded


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    允许不同模态之间进行信息交互
    """
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len_q, embed_dim)
            key: (batch, seq_len_k, embed_dim)
            value: (batch, seq_len_v, embed_dim)
            key_padding_mask: (batch, seq_len_k) 掩码
        Returns:
            output: (batch, seq_len_q, embed_dim)
        """
        # 多头注意力
        attn_output, attn_weights = self.multihead_attn(
            query, key, value,
            key_padding_mask=key_padding_mask
        )
        
        # 残差连接 + 层归一化
        query = self.norm1(query + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(query)
        
        # 残差连接 + 层归一化
        output = self.norm2(query + ffn_output)
        
        return output


class GatedFusion(nn.Module):
    """
    门控融合机制
    动态学习各模态的重要性权重
    """
    
    def __init__(self, input_dim: int, num_modalities: int):
        """
        Args:
            input_dim: 每个模态的特征维度
            num_modalities: 模态数量
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_modalities = num_modalities
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * num_modalities, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU()
        )
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: 列表，每个元素为 (batch, input_dim)
        Returns:
            fused: (batch, input_dim) 融合后的特征
        """
        # 拼接所有模态
        concatenated = torch.cat(modality_features, dim=-1)  # (batch, input_dim * num_modalities)
        
        # 计算门控权重
        gate_weights = self.gate_network(concatenated)  # (batch, num_modalities)
        
        # 加权融合
        fused = torch.zeros_like(modality_features[0])
        for i, features in enumerate(modality_features):
            weight = gate_weights[:, i:i+1]  # (batch, 1)
            fused = fused + weight * features
        
        # 特征变换
        fused = self.feature_transform(fused)
        
        return fused


class AdvancedMultimodalFusion(nn.Module):
    """
    高级多模态融合模块
    整合模态嵌入、跨模态注意力和门控融合
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        embed_dim: int = 64,
        num_heads: int = 4,
        num_fusion_layers: int = 2,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: 各模态的输入维度
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_fusion_layers: 融合层数
            output_dim: 输出维度
            dropout: Dropout比率
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        
        # 模态嵌入
        self.modality_embedding = ModalityEmbedding(modality_dims, embed_dim)
        
        # 跨模态注意力层
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])
        
        # 门控融合
        self.gated_fusion = GatedFusion(embed_dim, self.num_modalities)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        modality_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            modality_features: 字典，键为模态名，值为特征张量 (batch, dim)
            modality_masks: 可选的模态掩码，指示哪些模态可用
        Returns:
            fused_features: (batch, output_dim) 融合后的特征
        """
        # 1. 模态嵌入
        embedded = self.modality_embedding(modality_features)
        
        # 2. 转换为序列格式用于注意力计算
        modality_names = list(embedded.keys())
        modality_tensors = [embedded[name].unsqueeze(1) for name in modality_names]
        modality_sequence = torch.cat(modality_tensors, dim=1)  # (batch, num_modalities, embed_dim)
        
        # 3. 跨模态注意力
        attended = modality_sequence
        for cross_modal_layer in self.cross_modal_layers:
            attended = cross_modal_layer(attended, attended, attended)
        
        # 4. 分离各模态
        attended_modalities = [attended[:, i, :] for i in range(self.num_modalities)]
        
        # 5. 门控融合
        fused = self.gated_fusion(attended_modalities)
        
        # 6. 输出投影
        output = self.output_projection(fused)
        
        return output


class AdaptiveModalityFusion(nn.Module):
    """
    自适应模态融合（处理模态缺失）
    """
    
    def __init__(self, modality_dims: Dict[str, int], output_dim: int = 256):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        
        # 为每个模态创建独立的编码器
        self.modality_encoders = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.modality_encoders[modality] = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, output_dim)
            )
        
        # 自适应权重网络
        self.adaptive_weights = nn.Sequential(
            nn.Linear(len(modality_dims), 64),
            nn.ReLU(),
            nn.Linear(64, len(modality_dims)),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self, 
        modality_features: Dict[str, torch.Tensor],
        available_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            modality_features: 模态特征字典
            available_mask: (batch, num_modalities) 可用性掩码
        Returns:
            fused: (batch, output_dim)
        """
        batch_size = next(iter(modality_features.values())).size(0)
        
        # 编码各模态
        encoded_modalities = []
        modality_names = list(self.modality_dims.keys())
        
        for modality in modality_names:
            if modality in modality_features:
                encoded = self.modality_encoders[modality](modality_features[modality])
            else:
                # 模态缺失，使用零向量
                encoded = torch.zeros(batch_size, self.output_dim, device=next(self.parameters()).device)
            encoded_modalities.append(encoded)
        
        # 堆叠
        stacked = torch.stack(encoded_modalities, dim=1)  # (batch, num_modalities, output_dim)
        
        # 计算自适应权重
        if available_mask is None:
            available_mask = torch.ones(batch_size, len(modality_names), device=stacked.device)
        
        weights = self.adaptive_weights(available_mask)  # (batch, num_modalities)
        weights = weights.unsqueeze(-1)  # (batch, num_modalities, 1)
        
        # 加权融合
        fused = (stacked * weights).sum(dim=1)  # (batch, output_dim)
        
        return fused


if __name__ == "__main__":
    # 测试代码
    print("Testing Fusion Modules...")
    
    # 定义模态维度
    modality_dims = {
        'time_series': 6,
        'text': 128,
        'image': 64,
        'table': 6
    }
    
    # 创建测试数据
    batch_size = 4
    test_features = {
        'time_series': torch.randn(batch_size, 6),
        'text': torch.randn(batch_size, 128),
        'image': torch.randn(batch_size, 64),
        'table': torch.randn(batch_size, 6)
    }
    
    # 测试高级融合
    fusion = AdvancedMultimodalFusion(modality_dims, embed_dim=64, output_dim=256)
    output = fusion(test_features)
    print(f"✅ Advanced Fusion output shape: {output.shape}")
    
    # 测试自适应融合
    adaptive_fusion = AdaptiveModalityFusion(modality_dims, output_dim=256)
    output = adaptive_fusion(test_features)
    print(f"✅ Adaptive Fusion output shape: {output.shape}")
    
    print("✅ All fusion tests passed!")

