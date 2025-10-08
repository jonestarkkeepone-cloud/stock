#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化脚本 - 用于生成训练和预测结果的可视化图表
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from configs.config import Config
from data.dataset import FinMultiTimeDataset
from data.dataloader import create_dataloaders
from models.tkan_model import MultimodalTKANModel
from utils.visualization import (
    plot_training_history,
    plot_predictions,
    create_comprehensive_report
)
from utils.metrics import calculate_mse, calculate_mae, calculate_rmse, calculate_r2
from tqdm import tqdm


def visualize_training_history(log_dir: str, output_dir: str):
    """可视化训练历史"""
    history_path = Path(log_dir) / 'training_history.json'
    if not history_path.exists():
        print(f"训练历史文件不存在: {history_path}")
        return
    
    output_path = Path(output_dir) / 'training_history.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_training_history(str(history_path), str(output_path))


def visualize_predictions(checkpoint_path: str, config: Config, output_dir: str):
    """可视化预测结果"""
    print("加载模型和数据...")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载数据集
    dataset = FinMultiTimeDataset(
        data_dir=config.data.data_dir,
        market=config.data.market,
        stocks=config.data.stocks,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        seq_length=config.data.seq_length,
        pred_horizon=config.data.pred_horizon,
        use_cnn_features=config.data.use_cnn_features,
        cnn_feature_dim=config.data.cnn_feature_dim
    )
    
    # 创建测试集
    _, _, test_loader = create_dataloaders(
        dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        batch_size=config.train.batch_size,
        seed=config.train.seed
    )
    
    # 加载模型
    model = MultimodalTKANModel(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        output_size=config.data.pred_horizon,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"在 {len(test_loader.dataset)} 个测试样本上进行预测...")
    
    # 收集预测
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='预测中'):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # 计算指标
    all_preds_tensor = torch.from_numpy(all_preds)
    all_targets_tensor = torch.from_numpy(all_targets)
    
    mse = calculate_mse(all_targets_tensor, all_preds_tensor)
    mae = calculate_mae(all_targets_tensor, all_preds_tensor)
    rmse = calculate_rmse(all_targets_tensor, all_preds_tensor)
    r2 = calculate_r2(all_targets_tensor, all_preds_tensor)
    
    print("\n测试集结果:")
    print("=" * 60)
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.6f}")
    print("=" * 60)
    
    # 绘制预测结果
    output_path = Path(output_dir) / 'predictions.png'
    plot_predictions(
        all_targets, all_preds,
        stock_name=f"{config.data.market} Stocks",
        save_path=str(output_path)
    )
    
    # 生成综合报告
    test_results = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'num_samples': len(test_loader.dataset),
        'config': {
            'market': config.data.market,
            'seq_length': config.data.seq_length,
            'pred_horizon': config.data.pred_horizon
        }
    }
    
    create_comprehensive_report(
        checkpoint_path=checkpoint_path,
        test_results=test_results,
        y_true=all_targets,
        y_pred=all_preds,
        output_dir=output_dir
    )


def main():
    parser = argparse.ArgumentParser(description='可视化多模态TKAN股票预测结果')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                       help='模型检查点路径')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志目录')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                       help='输出目录')
    parser.add_argument('--data_dir', type=str, default='./FinMultiTime',
                       help='数据集路径')
    parser.add_argument('--market', type=str, default='SP500',
                       choices=['SP500', 'HS300'],
                       help='市场选择')
    parser.add_argument('--stocks', type=str, nargs='+',
                       default=['AAPL', 'MSFT', 'TSLA'],
                       help='股票代码列表')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['history', 'predictions', 'all'],
                       help='可视化模式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("多模态TKAN股票预测 - 结果可视化")
    print("=" * 80 + "\n")
    
    # 初始化配置
    config = Config()
    config.data.data_dir = args.data_dir
    config.data.market = args.market
    config.data.stocks = args.stocks
    
    # 可视化训练历史
    if args.mode in ['history', 'all']:
        print("正在生成训练历史可视化...")
        visualize_training_history(args.log_dir, str(output_dir))
    
    # 可视化预测结果
    if args.mode in ['predictions', 'all']:
        print("\n正在生成预测结果可视化...")
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            visualize_predictions(str(checkpoint_path), config, str(output_dir))
        else:
            print(f"错误: 检查点文件不存在: {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print(f"可视化完成! 结果已保存至: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

