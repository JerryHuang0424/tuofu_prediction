# feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from config import Config
from data_loader import DataLoader

class FeatureEngineer:
    def __init__(self, config=None):
        self.config = config or Config()
        self.preprocessor = None
    
    def create_features(self, data):
        """
        创建机器学习特征
        """
        print("\n正在创建特征...")
        features = []
        
        for row in data:
            if len(row) < 3:  # 至少需要label, feature1, feature2
                continue
                
            label, f1, f2, *time_points_raw = row
            
            # 转换时间字符串为datetime对象
            time_points = []
            for tp in time_points_raw:
                try:
                    dt = pd.to_datetime(str(tp))
                    time_points.append(dt)
                except Exception as e:
                    print(f"跳过无效时间: {tp}，原因: {e}")
            
            time_points = sorted(time_points)
            
            if len(time_points) == 0:
                continue
            
            feature_row = self._create_single_sample_features(label, f1, f2, time_points)
            features.append(feature_row)
        
        feature_df = pd.DataFrame(features)
        print(f"特征创建完成! 形状: {feature_df.shape}")
        return feature_df
    
    def _create_single_sample_features(self, label, f1, f2, time_points):
        """
        为单个样本创建特征
        """
        feature_row = {
            'label': label,
            'feature1': f1,
            'feature2': f2,
            'num_occurrences': len(time_points),
            'total_duration': (time_points[-1] - time_points[0]).days if len(time_points) > 1 else 0,
        }
        
        # 时间间隔特征
        if len(time_points) > 1:
            intervals = [(time_points[i+1] - time_points[i]).days 
                        for i in range(len(time_points)-1)]
            feature_row.update({
                'avg_interval': np.mean(intervals),
                'std_interval': np.std(intervals) if len(intervals) > 1 else 0,
                'min_interval': min(intervals),
                'max_interval': max(intervals),
                'last_interval': intervals[-1] if intervals else 0,
            })
        else:
            feature_row.update({
                'avg_interval': 0, 'std_interval': 0, 'min_interval': 0, 
                'max_interval': 0, 'last_interval': 0,
            })
        
        # 时间趋势特征
        if len(time_points) > 2:
            days_from_start = [(tp - time_points[0]).days for tp in time_points]
            z = np.polyfit(range(len(days_from_start)), days_from_start, 1)
            z2 = np.polyfit(range(len(days_from_start)), days_from_start, 2)
            feature_row.update({
                'time_trend': z[0],
                'time_curvature': z2[0]
            })
        else:
            feature_row.update({
                'time_trend': 0,
                'time_curvature': 0
            })
        
        # 最近活动特征
        feature_row['days_since_last'] = (pd.Timestamp.now() - time_points[-1]).days
        feature_row['recency'] = 1 / (1 + feature_row['days_since_last'])
        
        # 时间特征
        feature_row['last_month'] = time_points[-1].month
        feature_row['last_dayofweek'] = time_points[-1].dayofweek
        
        return feature_row
    
    def create_target_variable(self, feature_df, future_date, prediction_period_days=None):
        """
        创建目标变量
        """
        future_date = future_date or self.config.FUTURE_DATE
        prediction_period_days = prediction_period_days or self.config.PREDICTION_PERIOD_DAYS
        print(f"\n创建目标变量: 预测在 {future_date} 之后 {prediction_period_days} 天内是否出现")
        
        # 这里使用启发式方法创建目标变量，实际应用中应使用真实标签
        target = []
        
        for idx, row in feature_df.iterrows():
            prob = self._calculate_occurrence_probability(row)
            target.append(1 if prob > 0.5 else 0)
        
        feature_df['target'] = target
        
        # 显示目标变量分布
        target_counts = feature_df['target'].value_counts()
        print("目标变量分布:")
        print(target_counts)
        if 1 in target_counts:
            print(f"正样本比例: {target_counts[1]/len(feature_df):.3f}")
        
        return feature_df
    
    def _calculate_occurrence_probability(self, row):
        """
        计算出现概率（启发式方法，需要根据实际业务调整）
        """
        # 业务规则：出现次数大于等于10次的样本，未来不会再出现
        if row['num_occurrences'] >= 10:
            return 0.0
        prob = 0.0
        # 出现次数越多，未来出现的可能性越大
        prob += min(row['num_occurrences'] * 0.1, 0.3)
        # 最近活跃的label更可能出现
        prob += 0.4 if row['days_since_last'] < 30 else 0.1
        # 间隔稳定的label更可能出现
        if row['std_interval'] > 0 and row['avg_interval'] > 0:
            prob += 0.2 if row['std_interval'] / row['avg_interval'] < 0.5 else 0
        # 添加随机因素
        prob += np.random.uniform(0, 0.3)
        return min(prob, 1.0)
    
    def create_preprocessor(self):
        """创建特征预处理器"""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        def cyclic_encoding(X):
            """
            周期性特征编码 - 修复版本
            X 是 pandas DataFrame，需要转换为 numpy 数组进行索引
            """
            if X.shape[1] == 0:
                return np.empty((len(X), 0))
            
            # 将 DataFrame 转换为 numpy 数组
            X_array = X.values if hasattr(X, 'values') else X
            
            encoded = []
            if 'last_month' in self.config.CYCLIC_FEATURES:
                # 获取 last_month 列的索引
                month_idx = self.config.CYCLIC_FEATURES.index('last_month')
                if month_idx < X_array.shape[1]:
                    month = X_array[:, month_idx]
                    month_sin = np.sin(2 * np.pi * month / 12)
                    month_cos = np.cos(2 * np.pi * month / 12)
                    encoded.extend([month_sin, month_cos])
            
            if 'last_dayofweek' in self.config.CYCLIC_FEATURES:
                # 获取 last_dayofweek 列的索引
                day_idx = self.config.CYCLIC_FEATURES.index('last_dayofweek')
                if day_idx < X_array.shape[1]:
                    dayofweek = X_array[:, day_idx]
                    day_sin = np.sin(2 * np.pi * dayofweek / 7)
                    day_cos = np.cos(2 * np.pi * dayofweek / 7)
                    encoded.extend([day_sin, day_cos])
            
            return np.column_stack(encoded) if encoded else np.empty((len(X), 0))
        
        cyclic_transformer = Pipeline(steps=[
            ('cyclic', FunctionTransformer(cyclic_encoding, validate=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config.NUMERIC_FEATURES),
                ('cat', categorical_transformer, self.config.CATEGORICAL_FEATURES),
                ('cyclic', cyclic_transformer, self.config.CYCLIC_FEATURES)
            ]
        )
        
        self.preprocessor = preprocessor
        return preprocessor

def test_feature_engineer():
    """测试特征工程"""
    print("=== 测试特征工程 ===")
    feature_engineer = FeatureEngineer()
    
    # 使用真实数据
    from data_loader import DataLoader
    data_loader = DataLoader()
    data, _ = data_loader.load_data_from_excel()
    if data is None or len(data) == 0:
        print("未能加载真实数据，请检查Excel文件路径和sheet设置。")
        return
    # 特征创建
    feature_df = feature_engineer.create_features(data)
    print(f"特征数据形状: {feature_df.shape}")
    print(f"特征列: {feature_df.columns.tolist()}")
    # 目标变量创建
    feature_df_with_target = feature_engineer.create_target_variable(feature_df, '2025-12-13')
    assert 'target' in feature_df_with_target.columns
    print(f"目标变量分布: {feature_df_with_target['target'].value_counts().to_dict()}")
    # 预处理器创建
    preprocessor = feature_engineer.create_preprocessor()
    assert preprocessor is not None
    print("✓ 预处理器创建成功")
    print("✓ 特征工程测试通过")

if __name__ == "__main__":
    test_feature_engineer()