# ============================================================================
# evaluate.py - 评估脚本
# ============================================================================
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from configs.config import Config
from data.dataset import FinMultiTimeDataset
from data.dataloader import create_dataloaders
from models.tkan_model import MultimodalTKANModel
from utils.metrics import calculate_mse, calculate_mae, calculate_rmse, calculate_r2
from utils.visualization import create_comprehensive_report

def evaluate_model(checkpoint_path, config=None, visualize=True):
    """评估训练好的模型"""
    
    # 加载检查点
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    
    if config is None:
        config = checkpoint.get('config', Config())
    
    # 加载数据集
    print("Loading dataset...")
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
    print("Loading model...")
    model = MultimodalTKANModel(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        output_size=config.data.pred_horizon,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device(config.train.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"\nEvaluating on {len(test_loader.dataset)} samples...")
    print("=" * 60)
    
    # 收集预测
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Evaluating'):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 计算指标
    mse = calculate_mse(all_targets, all_preds)
    mae = calculate_mae(all_targets, all_preds)
    rmse = calculate_rmse(all_targets, all_preds)
    r2 = calculate_r2(all_targets, all_preds)

    print("\nTest Set Results:")
    print("=" * 60)
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.6f}")
    print("=" * 60)

    # 保存结果
    results = {
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

    results_path = Path(checkpoint_path).parent / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # 生成可视化报告
    if visualize:
        print("\n生成可视化报告...")
        y_true_np = all_targets.numpy()
        y_pred_np = all_preds.numpy()

        output_dir = Path(checkpoint_path).parent / 'evaluation_results'
        create_comprehensive_report(
            checkpoint_path=checkpoint_path,
            test_results=results,
            y_true=y_true_np,
            y_pred=y_pred_np,
            output_dir=str(output_dir)
        )

    return results