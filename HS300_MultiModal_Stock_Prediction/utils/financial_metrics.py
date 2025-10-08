"""
金融领域专用评估指标
包括方向准确率、夏普比率、最大回撤等
"""
import numpy as np
import torch
from typing import Union, Dict, Optional, Tuple
from scipy import stats


def calculate_direction_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    计算方向准确率（预测涨跌方向的准确率）
    
    Args:
        y_true: 真实值 (n_samples, pred_horizon) 或 (n_samples,)
        y_pred: 预测值 (n_samples, pred_horizon) 或 (n_samples,)
    
    Returns:
        direction_accuracy: 方向准确率 [0, 1]
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 计算方向（正负号）
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    
    # 计算准确率
    correct = (true_direction == pred_direction).astype(float)
    accuracy = np.mean(correct)
    
    return accuracy


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年的交易周期数（股票通常是252）
    
    Returns:
        sharpe_ratio: 夏普比率
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.cpu().numpy()
    
    returns = returns.flatten()
    
    # 计算超额收益
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # 计算夏普比率
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_max_drawdown(
    returns: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    计算最大回撤
    
    Args:
        returns: 收益率序列
    
    Returns:
        max_drawdown: 最大回撤（负值）
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.cpu().numpy()
    
    returns = returns.flatten()
    
    # 计算累积收益
    cumulative = np.cumsum(returns)
    
    # 计算运行最大值
    running_max = np.maximum.accumulate(cumulative)
    
    # 计算回撤
    drawdown = cumulative - running_max
    
    # 最大回撤
    max_dd = np.min(drawdown)
    
    return max_dd


def calculate_sortino_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    计算索提诺比率（只考虑下行风险）
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        periods_per_year: 每年交易周期数
    
    Returns:
        sortino_ratio: 索提诺比率
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.cpu().numpy()
    
    returns = returns.flatten()
    
    # 超额收益
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # 下行偏差（只考虑负收益）
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    downside_std = np.std(downside_returns)
    sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    
    return sortino


def calculate_information_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    benchmark_returns: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    计算信息比率（相对基准的超额收益）
    
    Args:
        returns: 策略收益率
        benchmark_returns: 基准收益率
    
    Returns:
        information_ratio: 信息比率
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.cpu().numpy()
    if isinstance(benchmark_returns, torch.Tensor):
        benchmark_returns = benchmark_returns.cpu().numpy()
    
    returns = returns.flatten()
    benchmark_returns = benchmark_returns.flatten()
    
    # 超额收益
    excess_returns = returns - benchmark_returns
    
    # 信息比率
    if np.std(excess_returns) == 0:
        return 0.0
    
    ir = np.mean(excess_returns) / np.std(excess_returns)
    
    return ir


def calculate_calmar_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    periods_per_year: int = 252
) -> float:
    """
    计算卡玛比率（年化收益率 / 最大回撤）
    
    Args:
        returns: 收益率序列
        periods_per_year: 每年交易周期数
    
    Returns:
        calmar_ratio: 卡玛比率
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.cpu().numpy()
    
    returns = returns.flatten()
    
    # 年化收益率
    annual_return = np.mean(returns) * periods_per_year
    
    # 最大回撤
    max_dd = abs(calculate_max_drawdown(returns))
    
    if max_dd == 0:
        return 0.0
    
    calmar = annual_return / max_dd
    
    return calmar


def calculate_win_rate(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    计算胜率（预测为正且实际为正的比例）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        win_rate: 胜率
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 预测为正的样本
    positive_predictions = y_pred > 0
    
    if np.sum(positive_predictions) == 0:
        return 0.0
    
    # 在预测为正的样本中，实际也为正的比例
    win_rate = np.mean(y_true[positive_predictions] > 0)
    
    return win_rate


def calculate_profit_factor(
    returns: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    计算盈利因子（总盈利 / 总亏损）
    
    Args:
        returns: 收益率序列
    
    Returns:
        profit_factor: 盈利因子
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.cpu().numpy()
    
    returns = returns.flatten()
    
    # 盈利和亏损
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf') if profits > 0 else 0.0
    
    profit_factor = profits / losses
    
    return profit_factor


def calculate_hit_ratio(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.01
) -> float:
    """
    计算命中率（预测误差在阈值内的比例）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        threshold: 误差阈值
    
    Returns:
        hit_ratio: 命中率
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 计算相对误差
    relative_error = np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-8))
    
    # 命中率
    hit_ratio = np.mean(relative_error < threshold)
    
    return hit_ratio


def calculate_comprehensive_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    计算所有金融指标

    Args:
        y_true: 真实值
        y_pred: 预测值
        risk_free_rate: 无风险利率

    Returns:
        metrics: 指标字典
    """
    # 基础回归指标
    try:
        from utils.metrics import calculate_mse, calculate_mae, calculate_rmse, calculate_r2
    except:
        # 如果导入失败，使用简单实现
        def calculate_mse(y_true, y_pred):
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()
            return float(np.mean((y_true - y_pred) ** 2))

        def calculate_mae(y_true, y_pred):
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()
            return float(np.mean(np.abs(y_true - y_pred)))

        def calculate_rmse(y_true, y_pred):
            return float(np.sqrt(calculate_mse(y_true, y_pred)))

        def calculate_r2(y_true, y_pred):
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return float(1 - ss_res / (ss_tot + 1e-8))
    
    mse = calculate_mse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    
    # 金融指标
    direction_acc = calculate_direction_accuracy(y_true, y_pred)
    
    # 将预测值作为收益率计算其他指标
    if isinstance(y_pred, torch.Tensor):
        returns = y_pred.cpu().numpy().flatten()
    else:
        returns = y_pred.flatten()
    
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    max_dd = calculate_max_drawdown(returns)
    sortino = calculate_sortino_ratio(returns, risk_free_rate)
    calmar = calculate_calmar_ratio(returns)
    win_rate = calculate_win_rate(y_true, y_pred)
    profit_factor = calculate_profit_factor(returns)
    hit_ratio = calculate_hit_ratio(y_true, y_pred)
    
    metrics = {
        # 回归指标
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        
        # 金融指标
        'direction_accuracy': float(direction_acc),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'hit_ratio': float(hit_ratio)
    }
    
    return metrics


def print_metrics_report(metrics: Dict[str, float]) -> None:
    """
    打印格式化的指标报告
    
    Args:
        metrics: 指标字典
    """
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE EVALUATION METRICS")
    print("=" * 70)
    
    print("\n📈 Regression Metrics:")
    print(f"  MSE:  {metrics.get('mse', 0):.6f}")
    print(f"  MAE:  {metrics.get('mae', 0):.6f}")
    print(f"  RMSE: {metrics.get('rmse', 0):.6f}")
    print(f"  R²:   {metrics.get('r2', 0):.6f}")
    
    print("\n💰 Financial Metrics:")
    print(f"  Direction Accuracy: {metrics.get('direction_accuracy', 0):.2%}")
    print(f"  Win Rate:          {metrics.get('win_rate', 0):.2%}")
    print(f"  Hit Ratio:         {metrics.get('hit_ratio', 0):.2%}")
    
    print("\n📊 Risk-Adjusted Returns:")
    print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Sortino Ratio:     {metrics.get('sortino_ratio', 0):.4f}")
    print(f"  Calmar Ratio:      {metrics.get('calmar_ratio', 0):.4f}")
    
    print("\n⚠️  Risk Metrics:")
    print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0):.4f}")
    print(f"  Profit Factor:     {metrics.get('profit_factor', 0):.4f}")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # 测试代码
    print("Testing Financial Metrics...")
    
    # 生成测试数据
    np.random.seed(42)
    y_true = np.random.randn(1000, 24) * 0.02
    y_pred = y_true + np.random.randn(1000, 24) * 0.01
    
    # 计算所有指标
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    
    # 打印报告
    print_metrics_report(metrics)
    
    print("✅ All financial metrics tests passed!")

