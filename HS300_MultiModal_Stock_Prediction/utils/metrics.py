import torch
import numpy as np
from typing import Union

def calculate_mse(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """计算均方误差"""
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true).float()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float()
    return torch.mean((y_true - y_pred) ** 2).item()

def calculate_mae(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """计算平均绝对误差"""
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true).float()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float()
    return torch.mean(torch.abs(y_true - y_pred)).item()

def calculate_rmse(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """计算均方根误差"""
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true).float()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float()
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

def calculate_r2(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """计算R²决定系数"""
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true).float()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return (1 - ss_res / ss_tot).item()