import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict

class MultimodalPreprocessor:
    """多模态数据预处理器"""
    
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
    
    def fit_transform_time_series(self, data: np.ndarray, key: str) -> np.ndarray:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(data)
        self.scalers[key] = scaler
        return normalized
    
    def transform_time_series(self, data: np.ndarray, key: str) -> np.ndarray:
        if key not in self.scalers:
            raise ValueError(f"Scaler {key} not fitted")
        return self.scalers[key].transform(data)
    
    def fit_transform_fundamentals(self, data: np.ndarray, key: str) -> np.ndarray:
        if data.std() > 1e-6:
            scaler = StandardScaler()
            normalized = scaler.fit_transform(data)
            self.scalers[key] = scaler
            return normalized
        return data
    
    def inverse_transform(self, data: np.ndarray, key: str) -> np.ndarray:
        if key not in self.scalers:
            raise ValueError(f"Scaler {key} not fitted")
        return self.scalers[key].inverse_transform(data)