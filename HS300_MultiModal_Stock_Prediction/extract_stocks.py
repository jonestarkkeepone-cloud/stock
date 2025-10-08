#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取 HS300 中所有具有完整四模态数据的股票代码
"""

import pandas as pd
import json
from pathlib import Path

def extract_complete_stocks():
    """提取完整四模态股票"""
    
    # 读取数据描述文件
    desc_file = Path('/media/lr/data/shared_folder/multimodal_stock_prediction/Finmultime/hs300stock_data_description.csv')
    
    if not desc_file.exists():
        print(f"❌ 文件不存在: {desc_file}")
        return None
    
    # 读取CSV（使用GBK编码）
    df = pd.read_csv(desc_file, encoding='gbk')
    
    print("=" * 80)
    print("HS300 数据集分析")
    print("=" * 80)
    print(f"\n总股票数: {len(df)}")
    
    # 筛选完整四模态数据
    complete_stocks = df[
        (df['table'] == 1) & 
        (df['image'] == 1) & 
        (df['ts'] == 1) & 
        (df['text'] == 1)
    ]
    
    print(f"完整四模态股票数: {len(complete_stocks)}")
    
    # 按行业统计
    print("\n按行业分布:")
    industry_counts = complete_stocks['申万一级行业'].value_counts()
    for industry, count in industry_counts.head(10).items():
        print(f"  {industry}: {count} 个")
    
    # 提取股票代码（去掉.SS和.SZ后缀）
    stock_codes = complete_stocks['证券代码'].tolist()
    stock_codes_clean = [code.replace('.SS', '').replace('.SZ', '') for code in stock_codes]
    
    # 保存到JSON文件
    output_file = Path('hs300_complete_stocks.json')
    data = {
        'total_count': len(stock_codes_clean),
        'stock_codes': stock_codes_clean,
        'stock_info': complete_stocks[['证券代码', '证券简称', '申万一级行业']].to_dict('records')
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 股票代码已保存到: {output_file}")
    
    # 显示前20个股票
    print("\n前20个完整四模态股票:")
    print(complete_stocks[['证券代码', '证券简称', '申万一级行业']].head(20).to_string(index=False))
    
    # 生成配置文件示例
    print("\n" + "=" * 80)
    print("配置文件示例 (configs/config.py):")
    print("=" * 80)
    print(f"""
@dataclass
class DataConfig:
    \"\"\"数据配置\"\"\"
    data_dir: str = './Finmultime'
    market: str = 'HS300'
    stocks: List[str] = None  # 设置为None将自动加载所有完整四模态股票
    # 或者手动指定前N个股票进行测试:
    # stocks: List[str] = {stock_codes_clean[:10]}
    start_date: str = '2019-01-01'
    end_date: str = '2025-12-31'
    seq_length: int = 60
    pred_horizon: int = 24
    use_cnn_features: bool = True  # HS300有K线图像，启用CNN特征
    cnn_feature_dim: int = 64
    """)
    
    return stock_codes_clean

if __name__ == '__main__':
    stocks = extract_complete_stocks()
    
    if stocks:
        print("\n" + "=" * 80)
        print(f"✅ 成功提取 {len(stocks)} 个完整四模态股票!")
        print("=" * 80)
        print("\n下一步:")
        print("  1. 查看 hs300_complete_stocks.json 了解所有股票信息")
        print("  2. 运行 python update_config_hs300.py 更新配置文件")
        print("  3. 运行 python test_finmultime_data.py 测试数据加载")
        print("  4. 运行 python train.py --epochs 50 开始训练")
        print("=" * 80)

