
# ============================================================================
# data/dataset.py
# ============================================================================
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from utils.data_utils import load_time_series, load_news_sentiment, load_fundamentals
from utils.preprocessing import MultimodalPreprocessor

class FinMultiTimeDataset(Dataset):
    """多模态股票数据集（使用混合CNN提取K线特征）"""
    
    def __init__(self, data_dir, market, stocks, start_date, end_date,
                 seq_length, pred_horizon,
                 use_cnn_features=True, cnn_feature_dim=64,
                 preprocessor=None):

        self.base_data_dir = Path(data_dir)
        self.market = market

        # 适配Finmultime目录结构
        # 如果是Finmultime目录，使用特殊的路径结构
        if 'Finmultime' in str(data_dir):
            # Finmultime的目录结构不同，不需要添加market子目录
            self.data_dir = self.base_data_dir
            self.is_finmultime = True
        else:
            # FinMultiTime的标准结构: data_dir/market/
            self.data_dir = self.base_data_dir / market
            self.is_finmultime = False

        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.use_cnn_features = use_cnn_features
        
        if stocks is None:
            # 如果是Finmultime的HS300，尝试加载完整四模态股票列表
            if self.is_finmultime and self.market == 'HS300':
                stocks = self._load_complete_hs300_stocks()
                if not stocks:
                    # 如果加载失败，回退到自动发现
                    stocks = self._discover_stocks()
            else:
                stocks = self._discover_stocks()
        self.stocks = stocks
        
        self.preprocessor = preprocessor or MultimodalPreprocessor()
        
        # 初始化CNN编码器
        if use_cnn_features:
            from models.image_encoder import HybridKLineEncoder
            self.image_encoder = HybridKLineEncoder(feature_dim=cnn_feature_dim)
            self.image_encoder.eval()
            
            self.image_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        self.samples = []
        self._load_and_align_data()
    
    def _load_complete_hs300_stocks(self):
        """加载HS300完整四模态股票列表"""
        import json
        stock_file = Path('hs300_complete_stocks.json')

        if not stock_file.exists():
            print(f"⚠️  未找到 {stock_file}，将自动发现股票")
            return None

        try:
            with open(stock_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            stock_codes = data['stock_codes']
            print(f"✅ 加载了 {len(stock_codes)} 个完整四模态HS300股票")
            return stock_codes
        except Exception as e:
            print(f"⚠️  加载股票列表失败: {e}")
            return None

    def _discover_stocks(self):
        """自动发现可用的股票代码"""
        if self.is_finmultime:
            # Finmultime结构: time_series/S%26P500_time_series/S&P500_time_series/ 或 time_series/HS300_time_series/HS300_time_series/
            if self.market == 'SP500':
                # 注意：外层目录是 S%26P500_time_series (URL编码)，内层是 S&P500_time_series
                ts_dir = self.data_dir / 'time_series' / 'S%26P500_time_series' / 'S&P500_time_series'
            else:  # HS300
                ts_dir = self.data_dir / 'time_series' / 'HS300_time_series' / 'HS300_time_series'
        else:
            # FinMultiTime标准结构: market/time_series/
            ts_dir = self.data_dir / 'time_series'

        if not ts_dir.exists():
            raise ValueError(f"Directory not found: {ts_dir}")

        # 提取股票代码（去掉.SS/.SZ后缀）
        discovered = []
        for f in ts_dir.glob('*.csv'):
            # 文件名格式: 600000.SS.csv 或 000001.SZ.csv
            stock_code = f.stem.replace('.SS', '').replace('.SZ', '')
            discovered.append(stock_code)

        print(f"🔍 自动发现 {len(discovered)} 个股票，使用前10个进行测试")
        return discovered[:10]
    
    def _find_kline_image(self, stock, target_date):
        """查找K线图文件"""
        if self.is_finmultime:
            # Finmultime结构: image/HS300_image/HS300_image/ (只有HS300有图像)
            if self.market == 'HS300':
                img_dir = self.data_dir / 'image' / 'HS300_image' / 'HS300_image'
            else:
                # SP500在Finmultime中没有图像数据
                return None
        else:
            # FinMultiTime标准结构
            img_dir = self.data_dir / 'image'

        if not img_dir.exists():
            return None

        target_date = pd.to_datetime(target_date)
        year = target_date.year
        half = 'H1' if target_date.month <= 6 else 'H2'

        # 尝试多种命名格式
        for pattern in [f'{stock}_{year}{half}.png',
                       f'{stock}_{year}_{half}.png',
                       f'{stock}_{year}-{half}.png']:
            path = img_dir / pattern
            if path.exists():
                return path
        return None
    
    def _extract_image_features_cnn(self, stock, dates):
        """使用混合CNN提取K线特征"""
        if not self.use_cnn_features:
            return np.full((len(dates), 1), 0.5)
        
        features = []
        current_image_path = None
        current_features = None
        
        for date in dates:
            img_path = self._find_kline_image(stock, date)
            
            # 如果图像改变，重新提取特征
            if img_path and img_path != current_image_path:
                try:
                    img = Image.open(img_path).convert('L')
                    img_tensor = self.image_transform(img).unsqueeze(0)
                    
                    with torch.no_grad():
                        feat = self.image_encoder(img_tensor)
                    current_features = feat.squeeze(0).numpy()
                    current_image_path = img_path
                except:
                    current_features = np.zeros(self.image_encoder.output_projection[-1].out_features)
            
            if current_features is not None:
                features.append(current_features)
            else:
                features.append(np.zeros(64))  # 默认特征维度
        
        return np.array(features)
    
    def _get_stock_suffix(self, stock):
        """获取股票的市场后缀 (.SS 或 .SZ)"""
        if self.is_finmultime and self.market == 'HS300':
            # HS300股票：6开头是上海(.SS)，0/3开头是深圳(.SZ)
            if stock.startswith('6'):
                return '.SS'
            elif stock.startswith(('0', '3')):
                return '.SZ'
        return ''

    def _load_and_align_data(self):
        """加载并对齐所有模态"""
        failed_stocks = []

        for stock in self.stocks:
            try:
                print(f"Processing {stock}...")

                # 获取股票后缀
                suffix = self._get_stock_suffix(stock)
                stock_with_suffix = f"{stock}{suffix}"

                # 构建路径（适配Finmultime和FinMultiTime）
                if self.is_finmultime:
                    # Finmultime结构
                    if self.market == 'SP500':
                        # 注意：外层目录是 S%26P500_time_series (URL编码)，内层是 S&P500_time_series
                        ts_path = self.data_dir / 'time_series' / 'S%26P500_time_series' / 'S&P500_time_series' / f'{stock_with_suffix}.csv'
                        text_path = self.data_dir / 'text' / 'sp500_news' / 'sp500_news' / f'{stock_with_suffix}.jsonl'
                        table_path = self.data_dir / 'table' / 'SP500_tabular' / f'{stock_with_suffix}.json'
                    else:  # HS300
                        ts_path = self.data_dir / 'time_series' / 'HS300_time_series' / 'HS300_time_series' / f'{stock_with_suffix}.csv'
                        text_path = self.data_dir / 'text' / 'hs300news_summary' / 'hs300news_summary' / f'{stock_with_suffix}.jsonl'
                        table_path = self.data_dir / 'table' / 'hs300_tabular' / 'hs300_tabular' / f'{stock_with_suffix}' / 'income.jsonl'
                else:
                    # FinMultiTime标准结构
                    ts_path = self.data_dir / 'time_series' / f'{stock}.csv'
                    text_path = self.data_dir / 'text' / f'{stock}.jsonl'
                    table_path = self.data_dir / 'table' / f'{stock}.json'

                # 加载时序数据
                ts_result = load_time_series(ts_path, self.start_date, self.end_date)

                if ts_result is None:
                    continue

                ts_df, dates = ts_result

                if len(dates) < self.seq_length + self.pred_horizon:
                    continue

                ts_features = ts_df.drop('Date', axis=1).values

                # 加载其他模态
                news = load_news_sentiment(text_path, dates)

                # 使用CNN提取K线特征
                image_features = self._extract_image_features_cnn(stock, dates)

                fundamentals = load_fundamentals(table_path, dates)

                # 归一化
                ts_norm = self.preprocessor.fit_transform_time_series(
                    ts_features, f'{stock}_ts'
                )
                fund_norm = self.preprocessor.fit_transform_fundamentals(
                    fundamentals, f'{stock}_fund'
                )

                # 拼接: 6 + 1 + 64 + 6 = 77维
                multimodal_features = np.concatenate([
                    ts_norm,           # 6维
                    news,              # 1维
                    image_features,    # 64维
                    fund_norm          # 6维
                ], axis=1)

                # 创建滑动窗口样本
                for i in range(len(multimodal_features) - self.seq_length - self.pred_horizon + 1):
                    x = multimodal_features[i:i + self.seq_length]
                    y = ts_norm[i + self.seq_length:i + self.seq_length + self.pred_horizon, 3]

                    self.samples.append({
                        'stock': stock,
                        'x': torch.FloatTensor(x),
                        'y': torch.FloatTensor(y),
                        'date': dates[i + self.seq_length - 1],
                        'scaler_key': f'{stock}_ts'
                    })

                print(f"  Created {len([s for s in self.samples if s['stock']==stock])} samples")

            except Exception as e:
                print(f"  ⚠️  Failed to process {stock}: {str(e)}")
                failed_stocks.append(stock)
                continue

        # 打印总结
        if failed_stocks:
            print(f"\n⚠️  Failed to load {len(failed_stocks)} stocks: {failed_stocks[:10]}{'...' if len(failed_stocks) > 10 else ''}")
        print(f"✅ Successfully loaded {len(self.stocks) - len(failed_stocks)} stocks with {len(self.samples)} total samples\n")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]['x'], self.samples[idx]['y']


# ============================================================================
# data/__init__.py
# ============================================================================
# from .dataset import FinMultiTimeDataset
# from .dataloader import create_dataloaders
# __all__ = ['FinMultiTimeDataset', 'create_dataloaders']
