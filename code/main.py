# main.py
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from predictor import Predictor
from config import Config

def main():
    """主程序"""
    # 初始化配置和各个模块
    config = Config()
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)
    model_trainer = ModelTrainer(config)
    
    # 1. 加载数据
    data, columns = data_loader.load_data_from_excel()
    if data is None:
        print("无法加载数据，请检查文件路径")
        print("使用样本数据进行演示...")
        data, columns = data_loader.create_sample_data()
    
    # 2. 探索数据
    data_loader.explore_data(data, columns)
    
    # 3. 创建特征
    feature_df = feature_engineer.create_features(data)
    
    if len(feature_df) == 0:
        print("没有有效的特征数据，程序退出")
        return
    
    # 4. 创建目标变量
    feature_df = feature_engineer.create_target_variable(feature_df, config.FUTURE_DATE)
    
    # 5. 创建预处理器
    preprocessor = feature_engineer.create_preprocessor()
    
    # 6. 训练模型
    results = model_trainer.train_models(feature_df, preprocessor)
    
    # 7. 创建预测器并进行预测
    predictor = Predictor(model_trainer, feature_engineer)
    predictions = predictor.batch_predict(data, config.FUTURE_DATE)
    
    # 8. 保存结果
    predictor.save_predictions(predictions)
    
    print("\n=== 预测流程完成 ===")

def test_all_modules():
    """测试所有模块"""
    print("开始测试所有模块...")
    
    # 导入并运行各个模块的测试
    from config import test_config
    from data_loader import test_data_loader
    from feature_engineer import test_feature_engineer
    from model_trainer import test_model_trainer
    from predictor import test_predictor
    from utils import test_utils
    
    test_config()
    test_data_loader()
    test_feature_engineer()
    test_model_trainer()
    test_predictor()
    test_utils()
    
    print("\n🎉 所有模块测试完成!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 运行测试模式
        test_all_modules()
    else:
        # 运行主程序
        main()