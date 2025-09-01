"""
汽车燃油效率预测 - 使用CNN进行线性回归
Auto MPG数据集燃油效率预测项目
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AutoMPGCNNPredictor:
    """汽车燃油效率CNN预测器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.history = None
        
    def load_and_explore_data(self, csv_path='archive/auto-mpg.csv'):
        """加载和探索数据"""
        print("=== 数据加载和探索 ===")
        
        # 加载数据
        self.df = pd.read_csv(csv_path)
        print(f"数据形状: {self.df.shape}")
        print(f"数据列: {list(self.df.columns)}")
        
        # 基本信息
        print("\n数据基本信息:")
        print(self.df.info())
        
        # 统计描述
        print("\n数值特征统计描述:")
        print(self.df.describe())
        
        # 检查缺失值
        print("\n缺失值情况:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # 检查horsepower列中的'?'值
        if 'horsepower' in self.df.columns:
            horsepower_missing = (self.df['horsepower'] == '?').sum()
            print(f"horsepower列中'?'的数量: {horsepower_missing}")
        
        return self.df
    
    def visualize_data(self):
        """数据可视化"""
        print("\n=== 数据可视化 ===")
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Auto MPG 数据集探索性分析', fontsize=16)
        
        # MPG分布
        axes[0, 0].hist(self.df['mpg'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('MPG分布')
        axes[0, 0].set_xlabel('MPG')
        axes[0, 0].set_ylabel('频次')
        
        # 气缸数分布
        self.df['cylinders'].value_counts().plot(kind='bar', ax=axes[0, 1], color='lightgreen')
        axes[0, 1].set_title('气缸数分布')
        axes[0, 1].set_xlabel('气缸数')
        axes[0, 1].set_ylabel('数量')
        axes[0, 1].tick_params(axis='x', rotation=0)
        
        # 原产地分布
        origin_labels = {1: 'USA', 2: 'Europe', 3: 'Japan'}
        origin_counts = self.df['origin'].map(origin_labels).value_counts()
        axes[0, 2].pie(origin_counts.values, labels=origin_counts.index, autopct='%1.1f%%')
        axes[0, 2].set_title('原产地分布')
        
        # MPG vs 重量散点图
        axes[1, 0].scatter(self.df['weight'], self.df['mpg'], alpha=0.6, color='coral')
        axes[1, 0].set_title('MPG vs 重量')
        axes[1, 0].set_xlabel('重量')
        axes[1, 0].set_ylabel('MPG')
        
        # MPG vs 排量散点图
        axes[1, 1].scatter(self.df['displacement'], self.df['mpg'], alpha=0.6, color='gold')
        axes[1, 1].set_title('MPG vs 排量')
        axes[1, 1].set_xlabel('排量')
        axes[1, 1].set_ylabel('MPG')
        
        # 年份vs MPG
        year_mpg = self.df.groupby('model year')['mpg'].mean()
        axes[1, 2].plot(year_mpg.index, year_mpg.values, marker='o', color='purple')
        axes[1, 2].set_title('年份 vs 平均MPG')
        axes[1, 2].set_xlabel('年份')
        axes[1, 2].set_ylabel('平均MPG')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 相关性热力图
        plt.figure(figsize=(10, 8))
        # 只选择数值列进行相关性分析
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n=== 数据预处理 ===")
        
        # 处理horsepower列中的'?'值
        if 'horsepower' in self.df.columns:
            # 将'?'替换为NaN
            self.df['horsepower'] = self.df['horsepower'].replace('?', np.nan)
            # 转换为数值类型
            self.df['horsepower'] = pd.to_numeric(self.df['horsepower'])
            
            # 用中位数填充缺失值
            median_horsepower = self.df['horsepower'].median()
            self.df['horsepower'].fillna(median_horsepower, inplace=True)
            print(f"horsepower缺失值已用中位数 {median_horsepower} 填充")
        
        # 删除车名列（对预测无用）
        if 'car name' in self.df.columns:
            self.df = self.df.drop('car name', axis=1)
        
        # 准备特征和目标变量
        X = self.df.drop('mpg', axis=1)
        y = self.df['mpg']
        
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"特征列: {list(X.columns)}")
        
        return X, y
    
    def prepare_features(self, X, y):
        """特征工程"""
        print("\n=== 特征工程 ===")
        
        # 分离数值和分类特征
        numeric_features = ['displacement', 'horsepower', 'weight', 'acceleration']
        categorical_features = ['cylinders', 'model year', 'origin']
        
        # 标准化数值特征
        X_numeric = self.scaler.fit_transform(X[numeric_features])
        
        # 编码分类特征
        X_categorical = []
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
            encoded = self.label_encoders[feature].fit_transform(X[feature])
            X_categorical.append(encoded.reshape(-1, 1))
        
        X_categorical = np.hstack(X_categorical)
        
        # 合并所有特征
        X_processed = np.hstack([X_numeric, X_categorical])
        
        print(f"处理后特征形状: {X_processed.shape}")
        
        return X_processed, y.values
    
    def build_cnn_model(self, input_shape):
        """构建CNN模型"""
        print("\n=== 构建CNN模型 ===")
        
        model = keras.Sequential([
            # 将1D特征重塑为适合CNN的形状
            layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
            
            # 第一个卷积层
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # 第二个卷积层
            layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # 第三个卷积层
            layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            # 全局平均池化
            layers.GlobalAveragePooling1D(),
            
            # 全连接层
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # 输出层（回归任务）
            layers.Dense(1)
        ])
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(model.summary())
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=200):
        """训练模型"""
        print("\n=== 模型训练 ===")
        
        # 构建模型
        self.model = self.build_cnn_model(X_train.shape[1])
        
        # 设置回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """评估模型"""
        print("\n=== 模型评估 ===")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred = y_pred.flatten()
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"测试集性能:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        return y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可绘制")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(self.history.history['loss'], label='训练损失', color='blue')
        axes[0].plot(self.history.history['val_loss'], label='验证损失', color='red')
        axes[0].set_title('模型损失')
        axes[0].set_xlabel('轮次')
        axes[0].set_ylabel('损失')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE曲线
        axes[1].plot(self.history.history['mae'], label='训练MAE', color='blue')
        axes[1].plot(self.history.history['val_mae'], label='验证MAE', color='red')
        axes[1].set_title('模型MAE')
        axes[1].set_xlabel('轮次')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, y_true, y_pred):
        """绘制预测结果"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 真实值vs预测值散点图
        axes[0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2)
        axes[0].set_xlabel('真实MPG')
        axes[0].set_ylabel('预测MPG')
        axes[0].set_title('真实值 vs 预测值')
        axes[0].grid(True)
        
        # 残差图
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[1].axhline(y=0, color='red', linestyle='--')
        axes[1].set_xlabel('预测MPG')
        axes[1].set_ylabel('残差')
        axes[1].set_title('残差图')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    print("汽车燃油效率预测 - CNN回归模型")
    print("=" * 50)
    
    # 创建预测器实例
    predictor = AutoMPGCNNPredictor()
    
    # 1. 数据加载和探索
    df = predictor.load_and_explore_data()
    
    # 2. 数据可视化
    predictor.visualize_data()
    
    # 3. 数据预处理
    X, y = predictor.preprocess_data()
    
    # 4. 特征工程
    X_processed, y_processed = predictor.prepare_features(X, y)
    
    # 5. 数据分割
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    print(f"\n数据分割:")
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 6. 模型训练
    history = predictor.train_model(X_train, y_train, X_val, y_val)
    
    # 7. 模型评估
    y_pred, metrics = predictor.evaluate_model(X_test, y_test)
    
    # 8. 结果可视化
    predictor.plot_training_history()
    predictor.plot_predictions(y_test, y_pred)
    
    print("\n=== 项目完成 ===")
    print("生成的文件:")
    print("- data_exploration.png: 数据探索图")
    print("- correlation_heatmap.png: 相关性热力图")
    print("- training_history.png: 训练历史图")
    print("- prediction_results.png: 预测结果图")


if __name__ == "__main__":
    main()
