import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional

def load_time_series(file_path: Path, start_date: pd.Timestamp,
                     end_date: pd.Timestamp) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    """加载时序数据"""
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)

    # 转换Date列为datetime，并移除时区信息
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['Date'] = df['Date'].dt.tz_localize(None)

    # 确保start_date和end_date也没有时区信息
    if hasattr(start_date, 'tz') and start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    if hasattr(end_date, 'tz') and end_date.tz is not None:
        end_date = end_date.tz_localize(None)

    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    df = df.sort_values('Date').reset_index(drop=True)
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df['Returns'] = df['Close'].pct_change().fillna(0)
    feature_cols.append('Returns')
    
    return df[['Date'] + feature_cols], df['Date'].values

def load_news_sentiment(file_path: Path, dates: np.ndarray) -> np.ndarray:
    """加载新闻情感"""
    if not file_path.exists():
        return np.full((len(dates), 1), 0.5)
    
    sentiment_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                date = pd.to_datetime(item['Date']).date()
                score = item.get('LSA Sum', item.get('sentiment_score', 3))
                if isinstance(score, (int, float)):
                    sentiment_map[date] = (score - 1) / 4.0
    except:
        pass
    
    # 前向填充
    sentiments = []
    last_sentiment = 0.5
    for date in dates:
        date_key = pd.to_datetime(date).date()
        if date_key in sentiment_map:
            last_sentiment = sentiment_map[date_key]
        sentiments.append(last_sentiment)
    
    return np.array(sentiments).reshape(-1, 1)

def load_fundamentals(file_path: Path, dates: np.ndarray) -> np.ndarray:
    """加载财务基本面"""
    if not file_path.exists():
        return np.zeros((len(dates), 6))
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            table_data = json.load(f)
    except:
        return np.zeros((len(dates), 6))
    
    fundamental_map = {}
    for item in table_data:
        if not isinstance(item, dict):
            continue
        
        period_end = pd.to_datetime(
            item.get('end', item.get('period_end', item.get('Date')))
        ).date()
        
        metrics = [
            item.get('NetIncome', item.get('net_profit', 0)),
            item.get('OperatingCashFlow', item.get('operating_cash_flow', 0)),
            item.get('StockholdersEquity', item.get('shareholders_equity', 0)),
            item.get('TotalAssets', item.get('total_assets', 0)),
            item.get('TotalLiabilities', item.get('total_liabilities', 0)),
            item.get('Revenue', item.get('revenue', 0)),
        ]
        
        metrics = [float(m) if m is not None else 0.0 for m in metrics]
        fundamental_map[period_end] = metrics
    
    # 前向填充
    fundamentals = []
    last_fundamental = [0.0] * 6
    for date in dates:
        date_key = pd.to_datetime(date).date()
        for fd in sorted(fundamental_map.keys()):
            if fd <= date_key:
                last_fundamental = fundamental_map[fd]
        fundamentals.append(last_fundamental)
    
    return np.array(fundamentals)