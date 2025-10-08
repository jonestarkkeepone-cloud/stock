
# ============================================================================
# data/dataloader.py
# ============================================================================
import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple

def create_dataloaders(dataset, train_ratio, val_ratio,
                       batch_size, seed=42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


print("""
代码生成完成！

使用方法:
1. 按照上述结构创建目录和文件
2. 确保TKAN已安装: pip install tkan
3. 运行训练: python train.py
4. 评估: python evaluate.py

特点:
- 使用混合架构CNN提取64维K线特征
- 输入维度: 77 (6+1+64+6)
- CNN包含Self-Attention机制
- 完整的端到端训练流程
""")