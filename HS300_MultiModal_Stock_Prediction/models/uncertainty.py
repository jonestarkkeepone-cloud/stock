"""
不确定性估计模块
提供预测置信区间和不确定性量化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class UncertaintyHead(nn.Module):
    """
    不确定性预测头
    同时输出均值和方差
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出维度（预测步长）
        """
        super().__init__()
        
        # 均值预测头
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim)
        )
        
        # 方差预测头
        self.var_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) 输入特征
        Returns:
            mean: (batch, output_dim) 预测均值
            var: (batch, output_dim) 预测方差（正值）
        """
        mean = self.mean_head(x)
        
        # 使用softplus确保方差为正
        log_var = self.var_head(x)
        var = F.softplus(log_var) + 1e-6
        
        return mean, var


class GaussianNLLLoss(nn.Module):
    """
    高斯负对数似然损失
    用于训练不确定性模型
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        pred_mean: torch.Tensor, 
        pred_var: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_mean: 预测均值
            pred_var: 预测方差
            target: 真实值
        Returns:
            loss: 负对数似然损失
        """
        # 负对数似然: 0.5 * (log(var) + (target - mean)^2 / var)
        loss = 0.5 * (torch.log(pred_var) + (target - pred_mean) ** 2 / pred_var)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MCDropoutModel(nn.Module):
    """
    Monte Carlo Dropout模型
    通过多次前向传播估计不确定性
    """
    
    def __init__(self, base_model: nn.Module, num_samples: int = 10):
        """
        Args:
            base_model: 基础模型
            num_samples: MC采样次数
        """
        super().__init__()
        self.base_model = base_model
        self.num_samples = num_samples
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False):
        """
        Args:
            x: 输入
            return_uncertainty: 是否返回不确定性
        Returns:
            如果return_uncertainty=False: 预测均值
            如果return_uncertainty=True: (均值, 方差)
        """
        if not return_uncertainty:
            return self.base_model(x)
        
        # 启用dropout进行MC采样
        self.base_model.train()
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = self.base_model(x)
                predictions.append(pred)
        
        # 恢复eval模式
        self.base_model.eval()
        
        # 计算均值和方差
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch, output_dim)
        mean = predictions.mean(dim=0)
        var = predictions.var(dim=0)
        
        return mean, var


class EnsembleUncertainty(nn.Module):
    """
    集成模型不确定性估计
    使用多个模型的预测分歧来估计不确定性
    """
    
    def __init__(self, models: list):
        """
        Args:
            models: 模型列表
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False):
        """
        Args:
            x: 输入
            return_uncertainty: 是否返回不确定性
        Returns:
            如果return_uncertainty=False: 预测均值
            如果return_uncertainty=True: (均值, 方差)
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (num_models, batch, output_dim)
        
        mean = predictions.mean(dim=0)
        
        if return_uncertainty:
            var = predictions.var(dim=0)
            return mean, var
        else:
            return mean


class ConfidenceCalibration(nn.Module):
    """
    置信度校准模块
    校准模型的不确定性估计
    """
    
    def __init__(self, num_bins: int = 10):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer('bin_boundaries', torch.linspace(0, 1, num_bins + 1))
        self.register_buffer('bin_counts', torch.zeros(num_bins))
        self.register_buffer('bin_accuracies', torch.zeros(num_bins))
    
    def update(self, confidences: torch.Tensor, accuracies: torch.Tensor):
        """
        更新校准统计
        
        Args:
            confidences: 置信度 (batch,)
            accuracies: 准确性 (batch,)
        """
        for i in range(self.num_bins):
            lower = self.bin_boundaries[i]
            upper = self.bin_boundaries[i + 1]
            
            mask = (confidences >= lower) & (confidences < upper)
            if mask.sum() > 0:
                self.bin_counts[i] += mask.sum()
                self.bin_accuracies[i] += accuracies[mask].sum()
    
    def get_calibration_error(self) -> float:
        """
        计算期望校准误差（ECE）
        """
        valid_bins = self.bin_counts > 0
        bin_confidences = (self.bin_boundaries[:-1] + self.bin_boundaries[1:]) / 2
        
        bin_acc = torch.zeros_like(self.bin_accuracies)
        bin_acc[valid_bins] = self.bin_accuracies[valid_bins] / self.bin_counts[valid_bins]
        
        weights = self.bin_counts / self.bin_counts.sum()
        ece = (weights * torch.abs(bin_acc - bin_confidences)).sum()
        
        return ece.item()


def calculate_prediction_intervals(
    mean: torch.Tensor,
    std: torch.Tensor,
    confidence_level: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算预测区间
    
    Args:
        mean: 预测均值
        std: 预测标准差
        confidence_level: 置信水平
    
    Returns:
        lower_bound: 下界
        upper_bound: 上界
    """
    from scipy import stats
    
    # 计算z分数
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # 计算区间
    lower_bound = mean - z_score * std
    upper_bound = mean + z_score * std
    
    return lower_bound, upper_bound


def evaluate_uncertainty_quality(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor
) -> dict:
    """
    评估不确定性估计的质量
    
    Args:
        predictions: 预测值
        uncertainties: 不确定性（标准差）
        targets: 真实值
    
    Returns:
        metrics: 不确定性质量指标
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(uncertainties, torch.Tensor):
        uncertainties = uncertainties.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # 计算误差
    errors = np.abs(predictions - targets)
    
    # 不确定性与误差的相关性
    correlation = np.corrcoef(uncertainties.flatten(), errors.flatten())[0, 1]
    
    # 校准：误差是否在预测的不确定性范围内
    normalized_errors = errors / (uncertainties + 1e-8)
    calibration_score = np.mean(normalized_errors < 1.96)  # 95%置信区间
    
    # 锐度：不确定性的平均值（越小越好）
    sharpness = np.mean(uncertainties)
    
    metrics = {
        'uncertainty_correlation': float(correlation),
        'calibration_score': float(calibration_score),
        'sharpness': float(sharpness),
        'mean_uncertainty': float(np.mean(uncertainties)),
        'std_uncertainty': float(np.std(uncertainties))
    }
    
    return metrics


class BayesianLinear(nn.Module):
    """
    贝叶斯线性层
    使用变分推断估计权重的不确定性
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        # 权重均值和对数方差
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.weight_log_var = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        
        # 偏置均值和对数方差
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_var = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features)
        Returns:
            output: (batch, out_features)
        """
        # 采样权重
        weight_std = torch.exp(0.5 * self.weight_log_var)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
        
        # 采样偏置
        bias_std = torch.exp(0.5 * self.bias_log_var)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        
        # 线性变换
        output = F.linear(x, weight, bias)
        
        return output
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算KL散度（用于正则化）
        """
        # 权重的KL散度
        weight_kl = -0.5 * torch.sum(
            1 + self.weight_log_var - self.weight_mu.pow(2) - self.weight_log_var.exp()
        )
        
        # 偏置的KL散度
        bias_kl = -0.5 * torch.sum(
            1 + self.bias_log_var - self.bias_mu.pow(2) - self.bias_log_var.exp()
        )
        
        return weight_kl + bias_kl


if __name__ == "__main__":
    # 测试代码
    print("Testing Uncertainty Modules...")
    
    # 测试不确定性预测头
    uncertainty_head = UncertaintyHead(input_dim=128, output_dim=24)
    test_input = torch.randn(4, 128)
    mean, var = uncertainty_head(test_input)
    print(f"✅ Uncertainty head - Mean shape: {mean.shape}, Var shape: {var.shape}")
    
    # 测试高斯NLL损失
    loss_fn = GaussianNLLLoss()
    target = torch.randn(4, 24)
    loss = loss_fn(mean, var, target)
    print(f"✅ Gaussian NLL Loss: {loss.item():.4f}")
    
    # 测试不确定性质量评估
    predictions = torch.randn(100, 24)
    uncertainties = torch.rand(100, 24) * 0.5
    targets = torch.randn(100, 24)
    metrics = evaluate_uncertainty_quality(predictions, uncertainties, targets)
    print(f"✅ Uncertainty quality metrics: {metrics}")
    
    print("✅ All uncertainty tests passed!")

