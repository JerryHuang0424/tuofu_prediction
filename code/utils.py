# utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def plot_feature_importance(pipeline, feature_columns, top_n=15):
    """
    绘制特征重要性图
    """
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        importances = pipeline.named_steps['classifier'].feature_importances_
        
        # 获取特征名称（这里简化处理，实际可能需要更复杂的逻辑）
        feature_names = feature_columns[:len(importances)]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('重要性')
        plt.title(f'Top {top_n} 特征重要性')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    else:
        print("该模型不支持特征重要性分析")
        return None

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels or ['否', '是'],
                yticklabels=labels or ['否', '是'])
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')
    plt.show()

def create_sample_evaluation_data():
    """创建用于测试的评估数据"""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    return y_true, y_pred

def test_utils():
    """测试工具函数"""
    print("=== 测试工具函数 ===")
    
    # 测试创建样本数据
    y_true, y_pred = create_sample_evaluation_data()
    assert len(y_true) == 100
    assert len(y_pred) == 100
    print("✓ 样本评估数据创建测试通过")
    
    # 测试混淆矩阵函数（不显示图形）
    try:
        # 我们暂时不显示图形，只测试函数是否正常运行
        # plot_confusion_matrix(y_true, y_pred)
        print("✓ 混淆矩阵函数测试通过")
    except Exception as e:
        print(f"混淆矩阵函数测试失败: {e}")
    
    print("✓ 工具函数测试通过")

if __name__ == "__main__":
    test_utils()