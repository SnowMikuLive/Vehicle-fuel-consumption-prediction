"""
简单的汽车燃油效率预测演示
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_simple_model():
    """创建简化的CNN模型用于快速演示"""
    model = keras.Sequential([
        layers.Reshape((7, 1), input_shape=(7,)),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def quick_demo():
    """快速演示"""
    print("汽车燃油效率预测快速演示")
    print("=" * 40)
    
    # 加载数据
    df = pd.read_csv('archive/auto-mpg.csv')
    
    # 预处理
    df['horsepower'] = df['horsepower'].replace('?', np.nan)
    df['horsepower'] = pd.to_numeric(df['horsepower'])
    df['horsepower'].fillna(df['horsepower'].median(), inplace=True)
    df = df.drop('car name', axis=1)
    
    # 准备数据
    X = df.drop('mpg', axis=1)
    y = df['mpg']
    
    # 简单的特征处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 创建和训练模型
    model = create_simple_model()
    print("\n训练模型中...")
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # 评估
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n模型性能:")
    print(f"测试MAE: {test_mae:.3f}")
    print(f"测试RMSE: {np.sqrt(test_loss):.3f}")
    print(f"R²得分: {r2:.3f}")
    
    # 示例预测
    print(f"\n示例预测:")
    sample_indices = [0, 1, 2]
    for i in sample_indices:
        actual = y_test.iloc[i]
        predicted = y_pred[i][0]
        print(f"样本 {i+1}: 实际MPG={actual:.1f}, 预测MPG={predicted:.1f}")

if __name__ == "__main__":
    quick_demo()
