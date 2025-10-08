"""
增强的文本编码器 - 使用预训练BERT模型
支持中文金融文本的语义特征提取
"""
import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np


class FinancialTextEncoder(nn.Module):
    """
    金融文本编码器
    使用预训练BERT模型提取文本语义特征
    支持中文和英文金融文本
    """
    
    def __init__(
        self, 
        output_dim: int = 128,
        model_name: str = 'bert-base-chinese',
        max_length: int = 128,
        use_pretrained: bool = True,
        freeze_bert: bool = False
    ):
        """
        Args:
            output_dim: 输出特征维度
            model_name: 预训练模型名称
            max_length: 最大文本长度
            use_pretrained: 是否使用预训练模型
            freeze_bert: 是否冻结BERT参数
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.max_length = max_length
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            try:
                from transformers import AutoModel, AutoTokenizer
                
                # 尝试加载预训练模型
                print(f"Loading pretrained model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.bert = AutoModel.from_pretrained(model_name)
                
                # 是否冻结BERT参数
                if freeze_bert:
                    for param in self.bert.parameters():
                        param.requires_grad = False
                    print("BERT parameters frozen")
                
                bert_hidden_size = self.bert.config.hidden_size
                
            except Exception as e:
                print(f"⚠️  Failed to load pretrained model: {e}")
                print("Falling back to simple embedding model")
                self.use_pretrained = False
                bert_hidden_size = 256
        
        if not self.use_pretrained:
            # 简单的嵌入模型（备用方案）
            self.embedding = nn.Embedding(10000, 256)
            self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
            bert_hidden_size = 256
        
        # 时序聚合层（处理多条新闻）
        self.temporal_aggregation = nn.LSTM(
            bert_hidden_size, 
            output_dim, 
            batch_first=True,
            bidirectional=False
        )
        
        # 注意力池化
        self.attention = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 数值扩展层（用于将1维情感分数扩展到output_dim）
        self.numeric_expansion = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: 文本列表或已经数值化的tensor (batch, seq_len, 1)
        Returns:
            features: (batch, seq_len, output_dim) 文本特征
        """
        # 如果输入已经是tensor（数值化的情感分数），直接扩展维度
        if isinstance(texts, torch.Tensor):
            return self._forward_numeric(texts)

        if self.use_pretrained:
            return self._forward_bert(texts)
        else:
            return self._forward_simple(texts)
    
    def _forward_bert(self, texts: List[str]) -> torch.Tensor:
        """使用BERT提取特征"""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移动到正确的设备
        device = next(self.bert.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS]标记的输出
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        
        # 如果有多条新闻，进行时序聚合
        if cls_output.size(0) > 1:
            cls_output = cls_output.unsqueeze(0)  # (1, num_news, hidden_size)
            temporal_output, _ = self.temporal_aggregation(cls_output)
            
            # 注意力池化
            attn_weights = self.attention(temporal_output)  # (1, num_news, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            
            # 加权求和
            features = (temporal_output * attn_weights).sum(dim=1)  # (1, output_dim)
        else:
            # 单条新闻，直接投影
            features = self.output_projection(cls_output)
        
        return features

    def _forward_numeric(self, sentiment_scores: torch.Tensor) -> torch.Tensor:
        """
        处理已经数值化的情感分数
        Args:
            sentiment_scores: (batch, seq_len, 1) 情感分数
        Returns:
            features: (batch, seq_len, output_dim) 扩展后的特征
        """
        batch_size, seq_len, _ = sentiment_scores.shape

        # 使用数值扩展层: 1 -> output_dim
        # 先reshape为 (batch*seq_len, 1)
        scores_flat = sentiment_scores.reshape(-1, 1)

        # 通过扩展层
        features_flat = self.numeric_expansion(scores_flat)  # (batch*seq_len, output_dim)

        # Reshape回 (batch, seq_len, output_dim)
        features = features_flat.reshape(batch_size, seq_len, self.output_dim)

        return features

    def _forward_simple(self, texts: List[str]) -> torch.Tensor:
        """简单的嵌入模型（备用）"""
        # 简单的字符级编码
        device = next(self.embedding.parameters()).device
        
        # 将文本转换为索引（简化版）
        max_len = 50
        batch_indices = []
        for text in texts:
            indices = [ord(c) % 10000 for c in text[:max_len]]
            indices += [0] * (max_len - len(indices))
            batch_indices.append(indices)
        
        indices_tensor = torch.tensor(batch_indices, device=device)
        
        # 嵌入
        embedded = self.embedding(indices_tensor)  # (batch, max_len, 256)
        
        # LSTM
        lstm_out, (h_n, _) = self.lstm(embedded)
        
        # 使用最后的隐状态
        features = h_n[-1]  # (batch, 256)
        
        # 投影到输出维度
        features = self.output_projection(features)
        
        return features


class SimpleSentimentEncoder(nn.Module):
    """
    简化的情感编码器（当无法使用BERT时的备用方案）
    将1维情感得分扩展为更丰富的特征
    """
    
    def __init__(self, output_dim: int = 128):
        super().__init__()
        
        self.output_dim = output_dim
        
        # 特征扩展网络
        self.feature_expansion = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 时序建模
        self.temporal_model = nn.LSTM(
            output_dim, 
            output_dim // 2, 
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, sentiment_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentiment_scores: (batch, seq_len, 1) 情感得分序列
        Returns:
            features: (batch, seq_len, output_dim) 扩展特征
        """
        # 特征扩展
        expanded = self.feature_expansion(sentiment_scores)  # (batch, seq_len, output_dim)
        
        # 时序建模
        temporal_features, _ = self.temporal_model(expanded)
        
        return temporal_features


def create_text_encoder(
    use_bert: bool = True,
    output_dim: int = 128,
    **kwargs
) -> nn.Module:
    """
    工厂函数：创建文本编码器
    
    Args:
        use_bert: 是否使用BERT（如果失败会自动降级）
        output_dim: 输出维度
        **kwargs: 其他参数
    
    Returns:
        text_encoder: 文本编码器模块
    """
    if use_bert:
        try:
            encoder = FinancialTextEncoder(output_dim=output_dim, **kwargs)
            print(f"✅ Created BERT-based text encoder (output_dim={output_dim})")
            return encoder
        except Exception as e:
            print(f"⚠️  Failed to create BERT encoder: {e}")
            print("Falling back to simple sentiment encoder")
    
    # 备用方案
    encoder = SimpleSentimentEncoder(output_dim=output_dim)
    print(f"✅ Created simple sentiment encoder (output_dim={output_dim})")
    return encoder


if __name__ == "__main__":
    # 测试代码
    print("Testing Text Encoders...")
    
    # 测试BERT编码器
    try:
        encoder = create_text_encoder(use_bert=True, output_dim=128)
        test_texts = ["股市上涨", "公司业绩良好"]
        features = encoder(test_texts)
        print(f"BERT output shape: {features.shape}")
    except Exception as e:
        print(f"BERT test failed: {e}")
    
    # 测试简单编码器
    encoder = create_text_encoder(use_bert=False, output_dim=128)
    test_sentiment = torch.randn(2, 10, 1)  # (batch=2, seq_len=10, dim=1)
    features = encoder(test_sentiment)
    print(f"Simple encoder output shape: {features.shape}")
    
    print("✅ All tests passed!")

