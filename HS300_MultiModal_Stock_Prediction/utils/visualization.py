# ============================================================================
# utils/visualization.py - 结果可视化工具
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import json
from typing import List, Dict, Optional
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_history(history_path: str, save_path: Optional[str] = None):
    """
    绘制训练历史曲线
    
    Args:
        history_path: 训练历史JSON文件路径
        save_path: 保存图片的路径（可选）
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 损失对比（对数尺度）
    plt.subplot(1, 2, 2)
    plt.semilogy(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.semilogy(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Loss Curves (Log Scale)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存至: {save_path}")
    
    plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    stock_name: str = "Stock",
                    save_path: Optional[str] = None,
                    max_samples: int = 500):
    """
    绘制预测结果对比图
    
    Args:
        y_true: 真实值 (n_samples, pred_horizon)
        y_pred: 预测值 (n_samples, pred_horizon)
        stock_name: 股票名称
        save_path: 保存路径
        max_samples: 最大显示样本数
    """
    # 限制显示样本数
    if len(y_true) > max_samples:
        indices = np.linspace(0, len(y_true) - 1, max_samples, dtype=int)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
    
    n_samples, pred_horizon = y_true.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. 预测vs真实值散点图（第一个时间步）
    ax = axes[0, 0]
    ax.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, s=20)
    min_val = min(y_true[:, 0].min(), y_pred[:, 0].min())
    max_val = max(y_true[:, 0].max(), y_pred[:, 0].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'{stock_name} - Prediction vs True (t+1)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 时间序列对比（前100个样本）
    ax = axes[0, 1]
    n_show = min(100, n_samples)
    x_axis = range(n_show)
    ax.plot(x_axis, y_true[:n_show, 0], 'b-', label='True', linewidth=1.5, alpha=0.7)
    ax.plot(x_axis, y_pred[:n_show, 0], 'r-', label='Predicted', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'{stock_name} - Time Series Comparison (First {n_show} samples)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. 误差分布直方图
    ax = axes[1, 0]
    errors = (y_pred - y_true).flatten()
    ax.hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{stock_name} - Error Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 不同预测步长的MAE
    ax = axes[1, 1]
    mae_per_step = np.mean(np.abs(y_pred - y_true), axis=0)
    steps = range(1, pred_horizon + 1)
    ax.bar(steps, mae_per_step, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Prediction Horizon', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title(f'{stock_name} - MAE by Prediction Horizon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存至: {save_path}")
    
    plt.show()


def plot_multi_stock_comparison(results: Dict[str, Dict], save_path: Optional[str] = None):
    """
    绘制多个股票的性能对比
    
    Args:
        results: {stock_name: {'mse': ..., 'mae': ..., 'rmse': ..., 'r2': ...}}
        save_path: 保存路径
    """
    stocks = list(results.keys())
    metrics = ['MSE', 'MAE', 'RMSE', 'R²']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_key = metric.lower().replace('²', '2')
        values = [results[stock][metric_key] for stock in stocks]
        
        bars = ax.bar(range(len(stocks)), values, alpha=0.7, edgecolor='black')
        
        # 颜色编码
        colors = plt.cm.viridis(np.linspace(0, 1, len(stocks)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xticks(range(len(stocks)))
        ax.set_xticklabels(stocks, rotation=45, ha='right')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison Across Stocks', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多股票对比图已保存至: {save_path}")
    
    plt.show()


def plot_attention_heatmap(attention_weights: np.ndarray, 
                          save_path: Optional[str] = None):
    """
    绘制注意力权重热力图
    
    Args:
        attention_weights: 注意力权重矩阵 (seq_len, seq_len)
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(attention_weights, cmap='YlOrRd', annot=False, 
                cbar_kws={'label': 'Attention Weight'})
    
    plt.xlabel('Key Position', fontsize=12)
    plt.ylabel('Query Position', fontsize=12)
    plt.title('Attention Weight Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力热力图已保存至: {save_path}")
    
    plt.show()


def create_comprehensive_report(checkpoint_path: str, 
                               test_results: Dict,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               output_dir: str = './results'):
    """
    创建综合评估报告
    
    Args:
        checkpoint_path: 模型检查点路径
        test_results: 测试结果字典
        y_true: 真实值
        y_pred: 预测值
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 绘制训练历史
    history_path = Path(checkpoint_path).parent.parent / 'logs' / 'training_history.json'
    if history_path.exists():
        plot_training_history(
            str(history_path),
            save_path=str(output_dir / 'training_history.png')
        )
    
    # 2. 绘制预测结果
    plot_predictions(
        y_true, y_pred,
        stock_name="Multi-Stock",
        save_path=str(output_dir / 'predictions.png')
    )
    
    # 3. 保存文本报告
    report_path = output_dir / 'evaluation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("多模态TKAN股票预测 - 评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"模型路径: {checkpoint_path}\n")
        f.write(f"测试样本数: {test_results['num_samples']}\n\n")
        
        f.write("性能指标:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  MSE:  {test_results['mse']:.6f}\n")
        f.write(f"  MAE:  {test_results['mae']:.6f}\n")
        f.write(f"  RMSE: {test_results['rmse']:.6f}\n")
        f.write(f"  R²:   {test_results['r2']:.6f}\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("配置信息:\n")
        f.write("-" * 80 + "\n")
        for key, value in test_results['config'].items():
            f.write(f"  {key}: {value}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n综合报告已生成至: {output_dir}")
    print(f"  - 训练历史图: training_history.png")
    print(f"  - 预测结果图: predictions.png")
    print(f"  - 文本报告: evaluation_report.txt")

