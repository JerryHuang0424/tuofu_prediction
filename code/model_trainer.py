# model_trainer.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from config import Config
from sklearn.pipeline import Pipeline

class ModelTrainer:
    def __init__(self, config=None):
        self.config = config or Config()
        self.pipeline = None
        self.feature_columns = None
        self.is_trained = False
    
    def train_models(self, feature_df, preprocessor):
        """训练多个机器学习模型"""
        print("\n=== 开始训练模型 ===")
        
        # 准备特征和目标
        X = feature_df.drop(columns=['label', 'target'])
        y = feature_df['target']
        
        self.feature_columns = X.columns.tolist()
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, 
            stratify=y
        )
        
        # 定义模型
        models = self._create_models(preprocessor)
        
        # 训练和评估模型
        results = {}
        for name, pipeline in models.items():
            print(f"训练 {name}...")
            pipeline.fit(X_train, y_train)
            
            # 评估
            train_score, test_score, auc_score = self._evaluate_model(pipeline, X_train, X_test, y_train, y_test)
            
            results[name] = {
                'pipeline': pipeline,
                'train_score': train_score,
                'test_score': test_score,
                'auc_score': auc_score
            }
            
            print(f"{name} - 训练准确率: {train_score:.3f}, 测试准确率: {test_score:.3f}, AUC: {auc_score:.3f}")
        
        # 选择最佳模型
        self.pipeline = self._select_best_model(results, X_test, y_test)
        self.is_trained = True
        
        return results
    
    def _create_models(self, preprocessor):
        """创建模型管道"""
        models = {
            'RandomForest': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    **self.config.MODELS['RandomForest'],
                    random_state=self.config.RANDOM_STATE
                ))
            ]),
            'GradientBoosting': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(
                    **self.config.MODELS['GradientBoosting'],
                    random_state=self.config.RANDOM_STATE
                ))
            ]),
            'XGBoost': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', xgb.XGBClassifier(
                    **self.config.MODELS['XGBoost'],
                    random_state=self.config.RANDOM_STATE
                ))
            ])
        }
        return models
    
    def _evaluate_model(self, pipeline, X_train, X_test, y_train, y_test):
        """评估单个模型"""
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        return train_score, test_score, auc_score
    
    def _select_best_model(self, results, X_test, y_test):
        """选择最佳模型"""
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_score'])
        best_pipeline = results[best_model_name]['pipeline']
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"测试集准确率: {results[best_model_name]['test_score']:.3f}")
        print(f"AUC: {results[best_model_name]['auc_score']:.3f}")
        
        # 显示详细分类报告
        y_pred_best = best_pipeline.predict(X_test)
        print("\n最佳模型分类报告:")
        print(classification_report(y_test, y_pred_best))
        
        return best_pipeline