# MNIST 手写数字识别 (PyTorch)

这是一个基于 PyTorch 深度学习框架实现的手写数字识别项目。项目使用经典的 MNIST 数据集，包含数据加载、模型构建、训练、测试及预测的完整流程。

## ✨ 项目特性
- 使用 PyTorch 构建卷积神经网络 (CNN) / 全连接网络 (根据你的实际情况修改)
- 自动下载并处理 MNIST 数据集
- 支持模型训练与保存
- 支持载入模型进行单张图片预测

## 📁 项目结构
```text
mnist-handwritten-digit/
├── data/                   # 数据集存放目录 (自动生成)
├── models/                 # 训练好的模型权重存放目录 (可选)
├── mnist_train.py                # 模型训练脚本
├── mnist_test.py                 # 模型测试/评估脚本
├── mnist_predict.py              # 单张图片预测脚本
├── requirements.txt        # 项目依赖包列表
├── .gitignore              # Git 忽略文件配置
└── README.md               # 项目说明文档