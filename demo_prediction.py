"""
汽车燃油效率预测演示
使用训练好的CNN模型进行新车燃油效率预测
"""

import numpy as np
import pandas as pd
from auto_mpg_cnn_prediction import AutoMPGCNNPredictor

def demo_prediction():
    """演示预测功能"""
    print("=== 汽车燃油效率预测演示 ===")
    
    # 创建预测器并训练模型
    predictor = AutoMPGCNNPredictor()
    
    # 加载和预处理数据
    df = predictor.load_and_explore_data()
    X, y = predictor.preprocess_data()
    X_processed, y_processed = predictor.prepare_features(X, y)
    
    # 简单分割数据进行训练
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # 快速训练模型（减少epoch数用于演示）
    predictor.train_model(X_train, y_train, X_val, y_val, epochs=50)
    
    # 演示预测
    print("\n=== 预测演示 ===")
    
    # 示例汽车数据
    example_cars = [
        {
            'cylinders': 4, 'displacement': 120, 'horsepower': 100,
            'weight': 2500, 'acceleration': 15, 'model year': 80, 'origin': 3,
            'description': '小型日系车'
        },
        {
            'cylinders': 8, 'displacement': 350, 'horsepower': 200,
            'weight': 4000, 'acceleration': 12, 'model year': 75, 'origin': 1,
            'description': '大型美系车'
        },
        {
            'cylinders': 6, 'displacement': 200, 'horsepower': 150,
            'weight': 3000, 'acceleration': 14, 'model year': 78, 'origin': 2,
            'description': '中型欧系车'
        }
    ]
    
    for i, car in enumerate(example_cars, 1):
        # 准备输入数据
        car_data = np.array([[
            car['cylinders'], car['displacement'], car['horsepower'],
            car['weight'], car['acceleration'], car['model year'], car['origin']
        ]])
        
        # 标准化数值特征
        car_numeric = predictor.scaler.transform(car_data[:, :4])  # 前4个是数值特征
        car_categorical = car_data[:, 4:]  # 后3个是分类特征
        
        # 对分类特征进行编码
        car_categorical_encoded = []
        categorical_features = ['acceleration', 'model year', 'origin']
        for j, feature in enumerate(categorical_features):
            if feature in predictor.label_encoders:
                # 确保值在训练范围内
                value = car_categorical[0, j]
                if feature == 'acceleration':
                    # acceleration是数值特征，不需要编码
                    continue
                elif feature == 'model year':
                    encoded_value = predictor.label_encoders[feature].transform([value])[0]
                elif feature == 'origin':
                    encoded_value = predictor.label_encoders[feature].transform([value])[0]
                car_categorical_encoded.append(encoded_value)
        
        # 重新组织特征
        # 数值特征：displacement, horsepower, weight, acceleration (标准化后)
        # 分类特征：cylinders, model year, origin (编码后)
        car_features = np.hstack([
            car_numeric,  # 4个标准化的数值特征
            [[predictor.label_encoders['cylinders'].transform([car['cylinders']])[0]]],
            [[predictor.label_encoders['model year'].transform([car['model year']])[0]]],
            [[predictor.label_encoders['origin'].transform([car['origin']])[0]]]
        ])
        
        # 预测
        predicted_mpg = predictor.model.predict(car_features, verbose=0)[0][0]
        
        print(f"\n车辆 {i}: {car['description']}")
        print(f"  气缸数: {car['cylinders']}")
        print(f"  排量: {car['displacement']}")
        print(f"  马力: {car['horsepower']}")
        print(f"  重量: {car['weight']}")
        print(f"  加速度: {car['acceleration']}")
        print(f"  年份: {car['model year']}")
        print(f"  原产地: {car['origin']} ({'美国' if car['origin']==1 else '欧洲' if car['origin']==2 else '日本'})")
        print(f"  预测MPG: {predicted_mpg:.2f}")


if __name__ == "__main__":
    demo_prediction()
