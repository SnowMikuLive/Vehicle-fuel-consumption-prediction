# 汽车燃油效率预测 - CNN回归模型

这个项目使用CNN（卷积神经网络）对Auto MPG数据集进行燃油效率预测。

## 项目特点

- 使用CNN进行线性回归任务
- 完整的数据探索和可视化
- 自动处理缺失值
- 特征工程和标准化
- 模型训练和验证
- 详细的性能评估和可视化

## 数据集信息

Auto MPG数据集包含398个汽车样本，具有以下特征：
- **mpg**: 燃油效率（目标变量）
- **cylinders**: 气缸数
- **displacement**: 排量
- **horsepower**: 马力
- **weight**: 重量
- **acceleration**: 加速度
- **model year**: 年份
- **origin**: 原产地（1=美国，2=欧洲，3=日本）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行项目

```bash
python auto_mpg_cnn_prediction.py
```

## 输出文件

运行后会生成以下可视化文件：
- `data_exploration.png`: 数据探索分析图
- `correlation_heatmap.png`: 特征相关性热力图
- `training_history.png`: 模型训练历史曲线
- `prediction_results.png`: 预测结果对比图

## CNN模型架构

模型采用1D CNN架构，包括：
- 3个卷积层（64, 32, 16个滤波器）
- 批量归一化和Dropout正则化
- 全局平均池化
- 2个全连接层
- 输出层（单个神经元，回归任务）

## 性能指标

模型使用以下指标评估：
- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数

## 项目结构

```
Vehicle fuel consumption prediction/
├── auto_mpg_cnn_prediction.py  # 主程序
├── requirements.txt             # 依赖包
├── README.md                   # 项目说明
├── archive/
│   └── auto-mpg.csv           # CSV格式数据
└── auto+mpg/
    ├── auto-mpg.data          # 原始数据
    ├── auto-mpg.names         # 数据说明
    └── ...
```

## 使用说明

1. 确保Python环境（建议3.7+）
2. 安装依赖包：`pip install -r requirements.txt`
3. 运行主程序：`python auto_mpg_cnn_prediction.py`
4. 查看生成的可视化图片了解结果

## 技术特点

- **数据预处理**: 自动处理缺失值，标准化数值特征
- **CNN架构**: 专为回归任务设计的1D CNN
- **正则化**: 使用Dropout和BatchNormalization防止过拟合
- **早停机制**: 自动停止训练避免过拟合
- **学习率调度**: 动态调整学习率优化训练
- **完整可视化**: 从数据探索到结果分析的全面可视化
