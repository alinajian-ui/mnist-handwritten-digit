
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# 1. 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)


# 2. 定义与训练时完全相同的模型结构
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


# 3. 加载训练好的模型参数
model = MNISTModel().to(device)

state_dict = torch.load(
    "mnist_model.pth",
    map_location=device,
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

print("模型加载成功！")


# 4. 加载 MNIST 测试集
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=False,
    transform=transforms.ToTensor()
)


# 5. 选取一张图片，index 可以修改为其他索引值来测试不同的图片
index = 520
image, true_label = test_dataset[index]

# 增加一个批次维度，并移动到运行设备
input_image = image.unsqueeze(0).to(device)


# 6. 进行预测
with torch.no_grad():
    output = model(input_image)
    probabilities = torch.softmax(output, dim=1)
    predicted_label = output.argmax(dim=1).item()
    confidence = probabilities[0, predicted_label].item() * 100


# 7. 输出预测结果
print("真实数字：", true_label)
print("预测数字：", predicted_label)
print(f"预测置信度：{confidence:.2f}%")


# 8. 显示图片
plt.imshow(image.squeeze(), cmap="gray")
plt.title(
    f"True: {true_label} | Predicted: {predicted_label}"
)
plt.axis("off")
plt.show()
