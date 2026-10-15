
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================
# 1. 配置运行设备
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备：", device)

if device.type == "cuda":
    print("显卡型号：", torch.cuda.get_device_name(0))


# =========================
# 2. 加载 MNIST 数据集
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)


# =========================
# 3. 定义神经网络
# =========================
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),             # 28×28 展平成 784 个数
            nn.Linear(28 * 28, 128),  # 输入层 -> 隐藏层
            nn.ReLU(),                # 激活函数，ReLU(x)=max(0,x)
            nn.Linear(128, 64),       # 隐藏层
            nn.ReLU(),
            nn.Linear(64, 10)         # 输出数字 0～9 的分数
        )

    def forward(self, x):
        return self.network(x)


model = MNISTModel().to(device)


# =========================
# 4. 定义损失函数和优化器
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 5. 训练一个轮次
# =========================
def train_one_epoch():
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # 将图片和标签放到 GPU（如果可用）
        images = images.to(device)
        labels = labels.to(device)

        # 清除上一步的梯度
        optimizer.zero_grad()

        # 前向传播：模型预测
        outputs = model(images)

        # 计算预测和正确标签之间的差距
        loss = criterion(outputs, labels)

        # 反向传播：计算梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total * 100

    return average_loss, accuracy


# =========================
# 6. 在测试集上评估
# =========================
def evaluate():
    model.eval()

    correct = 0
    total = 0

    # 测试阶段不需要计算梯度
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100


# =========================
# 7. 开始训练
# =========================
epochs = 5

for epoch in range(epochs):
    loss, train_accuracy = train_one_epoch()
    test_accuracy = evaluate()

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {loss:.4f} "
        f"训练准确率: {train_accuracy:.2f}% "
        f"测试准确率: {test_accuracy:.2f}%"
    )


# =========================
# 8. 保存训练好的模型
# =========================
torch.save(model.state_dict(), "mnist_model.pth")
print("训练完成！模型已保存到 mnist_model.pth")
