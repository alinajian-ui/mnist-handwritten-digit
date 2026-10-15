# MNIST 手写数字识别与 CNN 可视化

这是一个基于 **PyTorch** 和 **Tkinter** 的 MNIST 手写数字识别项目。项目不仅可以训练和测试数字识别模型，还提供交互式可视化界面，用于展示卷积神经网络从输入图像到最终分类结果的完整判断过程。

## 功能特性

- **手写数字识别**：在画布中书写 `0～9`，由训练好的 CNN 实时判断。
- **卷积过程可视化**：展示输入、卷积层和池化层的特征变化。
- **分类结果对比**：使用条形图比较 `0～9` 的预测概率。
- **判断过程说明**：显示前三名候选数字、置信度及第一名与第二名的概率差距。
- **模型训练与评估**：包含全连接网络和卷积神经网络的训练脚本。
- **CPU / CUDA 支持**：训练脚本会根据运行环境选择可用设备。

## 效果流程

```text
手写输入（1×28×28）
        ↓
卷积层 1 + ReLU（16×28×28）
        ↓
最大池化 1（16×14×14）
        ↓
卷积层 2 + ReLU（32×14×14）
        ↓
最大池化 2（32×7×7）
        ↓
全连接层
        ↓
Softmax 概率比较（0～9）
        ↓
预测数字与置信度
```

## 项目结构

```text
.
├── convolution_visualizer.py  # CNN 判断过程交互式可视化
├── mnist_app.py               # 基础手写数字识别界面
├── mnist_cnn.py               # CNN 模型训练与评估
├── mnist_train.py             # 全连接模型训练与评估
├── mnist_predict.py           # 使用测试集样本进行预测
├── mnist_cnn.pth              # 已训练的 CNN 权重
├── mnist_model.pth            # 已训练的全连接模型权重
├── requirements.txt           # Python 依赖
└── data/                      # MNIST 数据集目录
```

## 环境要求

- **Python**：建议使用 `3.10` 或更高版本
- **PyTorch**
- **torchvision**
- **Pillow**
- **Tkinter**：Windows 官方 Python 通常已自带

> [!NOTE]
> 项目目录已经包含 `.venv` 时，建议直接使用该虚拟环境。`convolution_visualizer.py` 在检测到系统 Python 未安装 PyTorch 时，会尝试自动切换到项目的 `.venv`。

## 安装

### 1. 创建虚拟环境

```powershell
python -m venv .venv
```

### 2. 激活虚拟环境

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止脚本执行，可在当前窗口临时修改执行策略：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如需运行 `mnist_predict.py` 的图片展示功能，还需安装 Matplotlib：

```powershell
pip install matplotlib
```

## 快速开始

### 启动 CNN 判断过程可视化

推荐使用新版可视化界面：

```powershell
python convolution_visualizer.py
```

也可以显式指定项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe convolution_visualizer.py
```

启动后：

1. 在左侧黑色画布中按住鼠标绘制数字。
2. 中间区域会实时展示每个卷积和池化阶段的特征图。
3. 右侧会比较 `0～9` 的预测概率。
4. 查看最终预测、最高置信度、前三名候选和概率差距。
5. 点击 **清空** 可重新绘制。

### 启动基础识别界面

```powershell
python mnist_app.py
```

该界面适合快速体验手写数字识别，不展示中间特征图。

## 模型训练

### 训练 CNN 模型

```powershell
python mnist_cnn.py
```

训练完成后会生成或覆盖 `mnist_cnn.pth`。此权重文件由 `convolution_visualizer.py` 和 `mnist_app.py` 使用。

### 训练全连接模型

```powershell
python mnist_train.py
```

训练完成后会生成或覆盖 `mnist_model.pth`。

### 测试全连接模型

```powershell
python mnist_predict.py
```

`mnist_predict.py` 会从本地 MNIST 测试集中读取样本，输出真实标签、预测标签和置信度，并使用 Matplotlib 显示图像。

## 可视化内容说明

| 区域 | 展示内容 | 作用 |
| --- | --- | --- |
| **输入区** | `28×28` 灰度手写图像 | 接收用户绘制的数字 |
| **卷积层 1** | `16` 个特征通道 | 提取基础边缘和笔画特征 |
| **池化层 1** | `16×14×14` 特征 | 压缩尺寸并保留显著响应 |
| **卷积层 2** | `32` 个特征通道 | 组合更复杂的数字结构特征 |
| **池化层 2** | `32×7×7` 特征 | 生成分类前的高层特征 |
| **概率对比区** | `0～9` 的 Softmax 概率 | 比较全部候选分类结果 |

特征图使用聚合方式显示多个通道：颜色越亮，表示该区域对当前层的响应越强。该图用于帮助理解模型关注的位置，并不等同于原始图像。

## 常见问题

### `import torch` 报错

出现以下错误：

```text
ModuleNotFoundError: No module named 'torch'
```

通常表示当前终端或编辑器使用了错误的 Python 解释器。请执行：

```powershell
.\.venv\Scripts\python.exe convolution_visualizer.py
```

在 VS Code 或其他 IDE 中，请将 Python 解释器切换为：

```text
项目目录\.venv\Scripts\python.exe
```

### 找不到模型文件

请确认 `mnist_cnn.pth` 与 `convolution_visualizer.py` 位于同一目录。若权重文件不存在，可运行：

```powershell
python mnist_cnn.py
```

### Tkinter 无法导入

Windows 官方 Python 一般自带 Tkinter。如果使用精简版 Python，请重新安装完整版本，并在安装时启用 **Tcl/Tk and IDLE**。

### 识别结果不准确

- 尽量将数字写在画布中央。
- 笔画不要过细或互相粘连。
- 尽量使用接近 MNIST 数据集的书写风格。
- 可重新训练模型或增加训练轮数以改善效果。

## 技术栈

- **PyTorch**：模型构建、训练与推理
- **torchvision**：MNIST 数据集加载和预处理
- **Tkinter**：桌面交互界面
- **Pillow**：基础识别界面的图像处理
- **Matplotlib**：测试样本展示

## 后续改进

- [ ] 增加单个卷积通道的切换和放大查看
- [ ] 支持导入本地数字图片
- [ ] 展示卷积核权重与单步卷积计算
- [ ] 增加训练损失和准确率曲线
- [ ] 导出预测结果与特征图

## 适用场景

本项目适合用于学习：

- MNIST 数据集的加载与预处理
- PyTorch 神经网络训练流程
- CNN 中卷积、激活和池化的作用
- Softmax 分类概率的含义
- 桌面端机器学习可视化界面的实现
