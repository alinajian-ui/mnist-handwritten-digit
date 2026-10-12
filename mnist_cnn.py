
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 1. 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备：", device)

if device.type == "cuda":
    print("显卡：", torch.cuda.get_device_name(0))


# 2. 加载数据集
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


# 3. 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # 输入：[批次, 1, 28, 28]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = CNN().to(device)


# 4. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 5. 训练一个轮次
def train_one_epoch():
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total * 100


# 6. 测试模型
def evaluate():
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100


# 7. 开始训练
epochs = 5

for epoch in range(epochs):
    loss, train_acc = train_one_epoch()
    test_acc = evaluate()

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {loss:.4f} "
        f"训练准确率: {train_acc:.2f}% "
        f"测试准确率: {test_acc:.2f}%"
    )


# 8. 保存 CNN 参数
torch.save(model.state_dict(), "mnist_cnn.pth")
print("CNN 训练完成，模型已保存到 mnist_cnn.pth")
