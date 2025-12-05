# data_loader.py
import pandas as pd
import numpy as np
import os
from config import Config

class DataLoader:
    def __init__(self, config=None):
        self.config = config or Config()
    
    def load_data_from_excel(self, file_path=None, sheet_name=None):
        """
        从Excel文件加载数据
        """
        file_path = file_path or self.config.EXCEL_FILE_PATH
        sheet_name = sheet_name or self.config.SHEET_NAME

        print(f"加载数据文件: {file_path}, 工作表: {sheet_name}")
        
        
        print("正在加载Excel文件...")
        try:
            df = pd.read_excel(file_path)
            print(df.head(5))
            print(f"数据加载成功! 形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            
            # 转换数据格式，并将所有时间字段统一为datetime.datetime类型
            import math
            from pandas import isna, NaT
            data = []
            for _, row in df.iterrows():
                row_data = []
                for col in df.columns:
                    val = row[col]
                    # 尝试将 pandas.Timestamp 转为 datetime.datetime
                    if isinstance(val, pd.Timestamp):
                        val = val.to_pydatetime()
                    elif isinstance(val, str):
                        try:
                            val = pd.to_datetime(val).to_pydatetime()
                        except Exception:
                            pass
                    # 过滤无效时间（nan、NaT、None）
                    if val is None or (isinstance(val, float) and math.isnan(val)) or val is NaT:
                        continue
                    row_data.append(val)
                data.append(tuple(row_data))
            return data, df.columns.tolist()
        
        except Exception as e:
            print(f"加载Excel文件时出错: {e}")
            return None, None
    
    def explore_data(self, data, columns):
        """
        探索数据基本情况
        """
        print("\n=== 数据探索 ===")
        print(f"总样本数: {len(data)}")
        
        if len(data) == 0:
            return {}
            
        # 统计特征分布
        feature1_values = set()
        feature2_values = set()
        time_counts = []
        
        for row in data:
            if len(row) < 3:  # 至少需要label, feature1, feature2
                continue
            label, f1, f2, *times = row
            feature1_values.add(f1)
            feature2_values.add(f2)
            time_counts.append(len(times))
        
        print(f"feature1 类别: {feature1_values}")
        print(f"feature2 类别: {feature2_values}")
        if time_counts:
            print(f"时间点数量统计 - 平均: {np.mean(time_counts):.2f}, 最小: {min(time_counts)}, 最大: {max(time_counts)}")
        
        # 显示前几个样本
        print("\n前3个样本示例:")
        for i in range(min(3, len(data))):
            print(f"样本 {i+1}: {data[i]}")
        
        return {
            'feature1_categories': list(feature1_values),
            'feature2_categories': list(feature2_values),
            'time_stats': {
                'mean': np.mean(time_counts) if time_counts else 0,
                'min': min(time_counts) if time_counts else 0,
                'max': max(time_counts) if time_counts else 0
            }
        }
    

def test_data_loader():
    """测试数据加载器"""
    print("=== 测试数据加载器 ===")
    data_loader = DataLoader()
    
    # 加载真实实验数据
    data, columns = data_loader.load_data_from_excel()
    if data is None or columns is None:
        print("未能加载真实数据，请检查Excel文件路径和sheet设置。")
        return
    print(f"实验数据数量: {len(data)}")
    # 数据探索
    exploration_result = data_loader.explore_data(data, columns)
    print(f"探索结果: {exploration_result}")
    # 简单数据格式检查
    assert len(columns) >= 3
    print("✓ 实验数据加载器测试通过")

if __name__ == "__main__":
    test_data_loader()