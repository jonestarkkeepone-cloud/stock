# ============================================================================
# train.py - 训练脚本
# ============================================================================
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import argparse
from tqdm import tqdm

from configs.config import Config
from data.dataset import FinMultiTimeDataset
from data.dataloader import create_dataloaders
from models.tkan_model import MultimodalTKANModel
from utils.preprocessing import MultimodalPreprocessor

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    losses = []
    
    pbar = tqdm(dataloader, desc='Training')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        losses.append(loss.item())
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return np.mean(losses)

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Validating'):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            losses.append(loss.item())
    
    return np.mean(losses)

def train_model(config: Config):
    """主训练函数"""
    
    # 设置随机种子
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)
    
    # 创建目录
    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config.train.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化预处理器
    preprocessor = MultimodalPreprocessor()
    
    # 加载数据集
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    dataset = FinMultiTimeDataset(
        data_dir=config.data.data_dir,
        market=config.data.market,
        stocks=config.data.stocks,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        seq_length=config.data.seq_length,
        pred_horizon=config.data.pred_horizon,
        use_cnn_features=config.data.use_cnn_features,
        cnn_feature_dim=config.data.cnn_feature_dim,
        preprocessor=preprocessor
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        batch_size=config.train.batch_size,
        seed=config.train.seed
    )
    
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")
    
    # 初始化模型
    model = MultimodalTKANModel(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        output_size=config.data.pred_horizon,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )
    
    device = torch.device(config.train.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input size: {config.model.input_size}")
    print(f"  Hidden size: {config.model.hidden_size}")
    print(f"  Output size: {config.data.pred_horizon}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}\n")
    
    # 设置训练
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(config.train.epochs):
        print(f"\nEpoch {epoch+1}/{config.train.epochs}")
        print("-" * 60)
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config
            }
            
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"✓ Best model saved (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config.train.patience}")
            
            if patience_counter >= config.train.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # 保存训练历史
    with open(log_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")
    
    return model, preprocessor
# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多模态TKAN股票预测')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate'],
                       help='运行模式: train 或 evaluate')
    parser.add_argument('--data_dir', type=str, default='./FinMultiTime',
                       help='数据集路径')
    parser.add_argument('--market', type=str, default='SP500',
                       choices=['SP500', 'HS300'],
                       help='市场选择')
    parser.add_argument('--stocks', type=str, nargs='+',     # ← 新增
                       default=['AAPL', 'MSFT', 'TSLA'],     # ← 新增
                       help='股票代码列表')                    # ← 新增
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                       help='检查点路径（用于evaluate模式）')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    
    args = parser.parse_args()
    
    # 初始化配置
    config = Config()
    config.data.data_dir = args.data_dir
    config.data.market = args.market
    config.data.stocks = args.stocks  # ← 命令行参数
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    
    if args.mode == 'train':
        print("\n" + "=" * 60)
        print("多模态TKAN股票预测 - 训练模式")
        print("=" * 60)
        print(f"市场: {config.data.market}")
        print(f"K线特征: 混合CNN (64维)")
        print(f"输入维度: {config.model.input_size} (6+1+64+6)")
        print("=" * 60 + "\n")
        
        train_model(config)
        
    elif args.mode == 'evaluate':
        print("\n" + "=" * 60)
        print("多模态TKAN股票预测 - 评估模式")
        print("=" * 60 + "\n")

        from evaluate import evaluate_model
        evaluate_model(args.checkpoint, config)