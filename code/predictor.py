# predictor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Predictor:
    def __init__(self, model_trainer, feature_engineer):
        self.model_trainer = model_trainer
        self.feature_engineer = feature_engineer
    
    def predict(self, label, feature1, feature2, historical_times, future_date):
        """预测特定label在未来时间点是否会出现"""
        if not self.model_trainer.is_trained:
            raise ValueError("模型尚未训练，请先调用train_models方法")
        
        # 创建特征
        time_points = [pd.to_datetime(str(tp)) for tp in historical_times if pd.notna(tp)]
        time_points = sorted(time_points)
        
        if len(time_points) == 0:
            return {
                'label': label,
                'error': '没有有效的时间数据'
            }
        
        feature_dict = self.feature_engineer._create_single_sample_features(label, feature1, feature2, time_points)
        feature_vector = self._create_feature_vector(feature_dict)
        
        # 预测
        probability = self.model_trainer.pipeline.predict_proba(feature_vector)[0][1]
        prediction = self.model_trainer.pipeline.predict(feature_vector)[0]
        
        return {
            'label': label,
            'feature1': feature1,
            'feature2': feature2,
            'future_date': future_date,
            'will_occur': bool(prediction),
            'probability': probability,
            'confidence': '高' if probability > 0.7 else '中' if probability > 0.5 else '低',
            'interpretation': f"基于历史模式，'{label}'在{future_date}出现的概率为{probability:.1%}"
        }
    
    def _create_feature_vector(self, feature_dict):
        """创建特征向量"""
        # 移除label和target（如果存在）
        feature_dict = {k: v for k, v in feature_dict.items() 
                       if k not in ['label', 'target']}
        
        feature_vector = pd.DataFrame([feature_dict])
        feature_vector = feature_vector.reindex(columns=self.model_trainer.feature_columns, fill_value=0)
        return feature_vector
    
    def batch_predict(self, data, future_date):
        """批量预测"""
        print(f"\n=== 开始批量预测 ===")
        predictions = []
        
        for row in data:
            if len(row) < 3:
                continue
                
            label, f1, f2, *times = row
            try:
                # 过滤掉空的时间值
                valid_times = [t for t in times if pd.notna(t)]
                if len(valid_times) == 0:
                    continue
                
                prediction = self.predict(label, f1, f2, valid_times, future_date)
                predictions.append(prediction)
                
            except Exception as e:
                print(f"预测 {label} 时出错: {e}")
                continue
        
        # 按概率排序
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        print(f"\n批量预测完成! 共预测 {len(predictions)} 个label")
        print("\n预测结果前10名:")
        for i, pred in enumerate(predictions[:30]):
            print(f"{i+1}. {pred['label']}: {pred['probability']:.1%} ({pred['confidence']}置信度)")
        
        return predictions
    
    def save_predictions(self, predictions, output_file=None):
        """保存预测结果"""
        output_file = output_file or self.feature_engineer.config.OUTPUT_FILE
        
        output_df = pd.DataFrame(predictions)
        output_df.to_excel(output_file, index=False)
        print(f"\n预测结果已保存到: {output_file}")
        return output_file

def test_predictor():
    """测试预测器"""
    print("=== 测试预测器 ===")
    
    # 创建必要的组件
    from model_trainer import ModelTrainer
    from feature_engineer import FeatureEngineer
    
    model_trainer = ModelTrainer()
    feature_engineer = FeatureEngineer()
    
    # 创建训练数据并训练模型
    sample_df = model_trainer.create_sample_training_data(50)
    preprocessor = feature_engineer.create_preprocessor()
    model_trainer.train_models(sample_df, preprocessor)
    
    # 创建预测器
    predictor = Predictor(model_trainer, feature_engineer)
    
    # 测试单个预测
    test_times = ['2023-01-01', '2023-01-15', '2023-02-01']
    prediction = predictor.predict('Test_Label', '难', 'animal', test_times, '2024-06-01')
    
    # 验证预测结果
    assert 'probability' in prediction
    assert 'confidence' in prediction
    assert 'will_occur' in prediction
    assert prediction['label'] == 'Test_Label'
    
    print(f"单个预测测试通过: {prediction}")
    
    # 测试批量预测
    test_data = [
        ('Label_1', '难', 'animal', '2023-01-01', '2023-01-15'),
        ('Label_2', '中', 'geology', '2023-02-01'),
        ('Label_3', '难', 'archeologist', '2023-01-10', '2023-02-01', '2023-03-01')
    ]
    
    batch_predictions = predictor.batch_predict(test_data, '2024-06-01')
    assert len(batch_predictions) > 0
    print(f"批量预测测试通过: 预测了 {len(batch_predictions)} 个样本")
    
    # 测试保存功能
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        output_file = predictor.save_predictions(batch_predictions, tmp.name)
        assert os.path.exists(output_file)
        os.unlink(output_file)  # 清理临时文件
    
    print("✓ 预测器测试通过")

if __name__ == "__main__":
    test_predictor()