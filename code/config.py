# config.py
class Config:
    # 文件配置
    EXCEL_FILE_PATH = "listening_con.xlsx"
    SHEET_NAME = 0
    OUTPUT_FILE = "prediction_listening_con_1115.xlsx"
    
    # 预测配置
    FUTURE_DATE = "2025-11-15"
    PREDICTION_PERIOD_DAYS = 0
    
    # 模型配置
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MODELS = {
        'RandomForest': {'n_estimators': 100},
        'GradientBoosting': {'n_estimators': 100},
        'XGBoost': {'n_estimators': 100, 'eval_metric': 'logloss'}
    }
    
    # 特征配置
    NUMERIC_FEATURES = [
        'num_occurrences', 'total_duration', 'avg_interval', 
        'std_interval', 'min_interval', 'max_interval', 'last_interval',
        'time_trend', 'time_curvature', 'days_since_last', 'recency'
    ]
    
    CATEGORICAL_FEATURES = ['feature1', 'feature2']
    CYCLIC_FEATURES = ['last_month', 'last_dayofweek']

def test_config():
    """测试配置类"""
    print("=== 测试配置类 ===")
    config = Config()
    
    # 测试配置项是否存在
    assert hasattr(config, 'EXCEL_FILE_PATH')
    assert hasattr(config, 'FUTURE_DATE')
    assert hasattr(config, 'NUMERIC_FEATURES')
    assert hasattr(config, 'CATEGORICAL_FEATURES')
    
    # 测试配置值类型
    assert isinstance(config.NUMERIC_FEATURES, list)
    assert isinstance(config.CATEGORICAL_FEATURES, list)
    assert isinstance(config.MODELS, dict)
    
    print("✓ 配置类测试通过")
    print(f"数值特征数量: {len(config.NUMERIC_FEATURES)}")
    print(f"分类特征数量: {len(config.CATEGORICAL_FEATURES)}")
    print(f"支持的模型: {list(config.MODELS.keys())}")

if __name__ == "__main__":
    test_config()