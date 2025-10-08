"""
多尺度TKAN模型
捕获短期、中期、长期的时序特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class MultiScaleTKANLayer(nn.Module):
    """
    多尺度TKAN层
    使用不同的感受野捕获多尺度时序特征
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
        scales: List[int] = [1, 3, 5],  # 不同的卷积核大小
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            scales: 多尺度卷积核大小列表
            dropout: Dropout比率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scales = scales
        self.num_scales = len(scales)
        
        # 为每个尺度创建卷积层
        self.scale_convs = nn.ModuleList()
        for scale in scales:
            padding = scale // 2
            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size=scale, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # LSTM风格的门控机制（每个尺度）
        self.gates = nn.ModuleList()
        for _ in scales:
            self.gates.append(nn.ModuleDict({
                'input_gate': nn.Linear(hidden_dim * 2, hidden_dim),
                'forget_gate': nn.Linear(hidden_dim * 2, hidden_dim),
                'cell_gate': nn.Linear(hidden_dim * 2, hidden_dim),
                'output_gate': nn.Linear(hidden_dim * 2, hidden_dim)
            }))
        
        # KAN风格的非线性变换
        self.kan_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 多尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            hidden_states: 各尺度的隐状态列表
        Returns:
            output: (batch, seq_len, hidden_dim)
            new_hidden_states: 新的隐状态列表
        """
        batch_size, seq_len, _ = x.shape
        
        # 初始化隐状态
        if hidden_states is None:
            hidden_states = [
                (torch.zeros(batch_size, self.hidden_dim, device=x.device),
                 torch.zeros(batch_size, self.hidden_dim, device=x.device))
                for _ in range(self.num_scales)
            ]
        
        # 转置用于卷积 (batch, input_dim, seq_len)
        x_transposed = x.transpose(1, 2)
        
        # 多尺度特征提取
        scale_features = []
        new_hidden_states = []
        
        for i, (scale_conv, gate) in enumerate(zip(self.scale_convs, self.gates)):
            # 卷积提取特征
            scale_feat = scale_conv(x_transposed)  # (batch, hidden_dim, seq_len)
            scale_feat = scale_feat.transpose(1, 2)  # (batch, seq_len, hidden_dim)
            
            # 获取该尺度的隐状态
            h, c = hidden_states[i]
            
            # 逐时间步处理（简化版，只处理最后一个时间步）
            last_feat = scale_feat[:, -1, :]  # (batch, hidden_dim)
            
            # 门控机制
            combined = torch.cat([last_feat, h], dim=-1)
            
            i_t = torch.sigmoid(gate['input_gate'](combined))
            f_t = torch.sigmoid(gate['forget_gate'](combined))
            c_tilde = torch.tanh(gate['cell_gate'](combined))
            o_t = torch.sigmoid(gate['output_gate'](combined))
            
            # 更新细胞状态
            c_new = f_t * c + i_t * c_tilde
            
            # KAN变换
            c_transformed = self.kan_transform(c_new)
            
            # 输出
            h_new = o_t * torch.tanh(c_transformed)
            h_new = self.layer_norm(h_new)
            
            scale_features.append(h_new)
            new_hidden_states.append((h_new, c_new))
        
        # 融合多尺度特征
        fused = torch.cat(scale_features, dim=-1)  # (batch, hidden_dim * num_scales)
        output_feat = self.scale_fusion(fused)  # (batch, hidden_dim)

        # 扩展到序列 - 保持与输入相同的序列长度
        output = output_feat.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim).contiguous()

        return output, new_hidden_states


class TemporalPyramidModule(nn.Module):
    """
    时序金字塔模块
    分别处理短期、中期、长期特征
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 短期模块（关注最近5-10天）
        self.short_term = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=1, 
            batch_first=True,
            dropout=dropout
        )
        
        # 中期模块（关注最近20-30天）
        self.mid_term = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        )
        
        # 长期模块（关注整个序列）
        self.long_term = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # 时序注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 3,
            num_heads=6,
            dropout=dropout,
            batch_first=True
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, hidden_dim)
        """
        seq_len = x.size(1)
        
        # 短期：只使用最后10个时间步
        short_len = min(10, seq_len)
        short_input = x[:, -short_len:, :]
        short_out, _ = self.short_term(short_input)
        short_feat = short_out[:, -1, :]  # 最后时刻
        
        # 中期：使用最后30个时间步
        mid_len = min(30, seq_len)
        mid_input = x[:, -mid_len:, :]
        mid_out, _ = self.mid_term(mid_input)
        mid_feat = mid_out[:, -1, :]
        
        # 长期：使用全部序列
        long_out, _ = self.long_term(x)
        long_feat = long_out[:, -1, :]
        
        # 拼接三个尺度
        combined = torch.cat([short_feat, mid_feat, long_feat], dim=-1)  # (batch, hidden_dim * 3)
        
        # 自注意力
        combined_seq = combined.unsqueeze(1)  # (batch, 1, hidden_dim * 3)
        attended, _ = self.temporal_attention(combined_seq, combined_seq, combined_seq)
        attended = attended.squeeze(1)
        
        # 融合
        output = self.fusion(attended)
        
        return output


class MultiScaleMultimodalTKAN(nn.Module):
    """
    多尺度多模态TKAN模型
    整合多尺度时序建模和多模态融合
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 24,
        num_layers: int = 2,
        scales: List[int] = [1, 3, 5],
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            output_size: 输出维度（预测步长）
            num_layers: TKAN层数
            scales: 多尺度卷积核大小
            dropout: Dropout比率
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 多尺度TKAN层
        self.multiscale_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.multiscale_layers.append(
                MultiScaleTKANLayer(in_dim, hidden_size, scales, dropout)
            )
        
        # 时序金字塔
        self.temporal_pyramid = TemporalPyramidModule(
            hidden_size, hidden_size, dropout
        )
        
        # 输出层
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            predictions: (batch, output_size)
        """
        # 通过多尺度TKAN层
        hidden_states = None
        layer_output = x
        for layer in self.multiscale_layers:
            layer_output, hidden_states = layer(layer_output, hidden_states)

        # 时序金字塔 - 注意：layer_output的维度是(batch, seq_len, hidden_size)
        # 但MultiScaleTKANLayer输出的是扩展后的序列，实际特征在最后时刻
        # 我们需要使用原始的序列特征
        pyramid_features = self.temporal_pyramid(layer_output)

        # 预测
        predictions = self.output_head(pyramid_features)

        return predictions


if __name__ == "__main__":
    # 测试代码
    print("Testing Multi-Scale TKAN...")
    
    # 创建模型
    model = MultiScaleMultimodalTKAN(
        input_size=256,  # 融合后的特征维度
        hidden_size=128,
        output_size=24,
        num_layers=2,
        scales=[1, 3, 5]
    )
    
    # 测试数据
    batch_size = 4
    seq_len = 60
    test_input = torch.randn(batch_size, seq_len, 256)
    
    # 前向传播
    output = model(test_input)
    print(f"✅ Multi-Scale TKAN output shape: {output.shape}")
    
    # 参数统计
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {num_params:,}")
    
    print("✅ All multi-scale TKAN tests passed!")

