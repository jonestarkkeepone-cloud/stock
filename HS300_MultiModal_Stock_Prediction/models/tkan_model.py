import torch
import torch.nn as nn
import torch.nn.functional as F


class TKANLayer(nn.Module):
    """PyTorch实现的TKAN层 - 基于KAN的时序网络层"""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # LSTM风格的门控机制
        self.W_i = nn.Linear(input_dim, hidden_dim)  # 输入门
        self.W_f = nn.Linear(input_dim, hidden_dim)  # 遗忘门
        self.W_c = nn.Linear(input_dim, hidden_dim)  # 候选值
        self.W_o = nn.Linear(input_dim, hidden_dim)  # 输出门

        # 循环连接
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # KAN风格的非线性变换（使用多项式基函数）
        self.kan_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: (batch, seq_len, input_dim) 或 (batch, input_dim)
            hidden_state: tuple of (h, c) 或 None
        Returns:
            output: (batch, seq_len, hidden_dim) 或 (batch, hidden_dim)
            (h, c): 新的隐状态
        """
        # 处理输入维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = x.shape

        # 初始化隐状态
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h, c = hidden_state

        outputs = []

        # 逐时间步处理
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)

            # 门控机制
            i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h))
            f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h))
            c_tilde = torch.tanh(self.W_c(x_t) + self.U_c(h))
            o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h))

            # 更新细胞状态
            c = f_t * c + i_t * c_tilde

            # KAN变换
            c_transformed = self.kan_transform(c)

            # 输出
            h = o_t * torch.tanh(c_transformed)
            h = self.layer_norm(h)
            h = self.dropout(h)

            outputs.append(h)

        # 堆叠输出
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)

        if squeeze_output:
            output = output.squeeze(1)  # (batch, hidden_dim)

        return output, (h, c)


class MultimodalTKANModel(nn.Module):
    """多模态TKAN预测模型"""
    
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers=2, dropout=0.2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 堆叠TKAN层
        self.tkan_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.tkan_layers.append(TKANLayer(in_dim, hidden_size))
        
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            predictions: (batch, output_size)
        """
        # 通过多层TKAN
        hidden_states = [None] * self.num_layers

        for layer_idx, tkan_layer in enumerate(self.tkan_layers):
            if layer_idx == 0:
                layer_input = x
            else:
                layer_input = layer_output

            # TKAN层处理整个序列
            layer_output, hidden_states[layer_idx] = tkan_layer(layer_input, hidden_states[layer_idx])
            layer_output = self.dropout(layer_output)

        # 使用最后时刻的隐状态
        last_hidden = layer_output[:, -1, :]  # (batch, hidden_size)

        # 预测
        predictions = self.fc_out(last_hidden)  # (batch, output_size)

        return predictions
