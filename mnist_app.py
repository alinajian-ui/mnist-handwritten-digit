
import tkinter as tk
from tkinter import messagebox

import torch
from torch import nn
from PIL import Image, ImageDraw, ImageOps


# =========================
# 1. 运行设备
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# 2. CNN 模型结构
# 必须与训练时一致
# =========================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
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


# =========================
# 3. 加载训练好的模型
# =========================
model = CNN().to(device)

try:
    state_dict = torch.load(
        "mnist_cnn.pth",
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    raise RuntimeError(
        "模型加载失败，请检查 mnist_cnn.pth 的路径，"
        "以及模型结构是否与训练时一致。"
    ) from e


# =========================
# 4. 创建手写画布
# =========================
root = tk.Tk()
root.title("MNIST 手写数字识别")
root.geometry("560x590")
root.resizable(False, False)

title = tk.Label(
    root,
    text="手写数字识别",
    font=("Microsoft YaHei", 20, "bold")
)
title.pack(pady=10)

description = tk.Label(
    root,
    text="请在黑色画布上写一个数字（0～9）",
    font=("Microsoft YaHei", 11)
)
description.pack(pady=4)

CANVAS_SIZE = 280

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
    cursor="cross"
)
canvas.pack(pady=8)

# 用同样大小的灰度图记录鼠标笔迹
image = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    0
)
draw = ImageDraw.Draw(image)

last_x = None
last_y = None
BRUSH_WIDTH = 20


# =========================
# 5. 鼠标书写功能
# =========================
def start_draw(event):
    global last_x, last_y
    last_x = event.x
    last_y = event.y

    # 画一个圆点，避免单击时没有笔迹
    r = BRUSH_WIDTH // 2

    canvas.create_oval(
        event.x - r, event.y - r,
        event.x + r, event.y + r,
        fill="white", outline="white"
    )

    draw.ellipse(
        (
            event.x - r, event.y - r,
            event.x + r, event.y + r
        ),
        fill=255
    )


def continue_draw(event):
    global last_x, last_y

    if last_x is not None and last_y is not None:
        canvas.create_line(
            last_x, last_y,
            event.x, event.y,
            fill="white",
            width=BRUSH_WIDTH,
            capstyle=tk.ROUND,
            smooth=True
        )

        draw.line(
            (last_x, last_y, event.x, event.y),
            fill=255,
            width=BRUSH_WIDTH
        )

        # 用圆形端点让笔迹更平滑
        r = BRUSH_WIDTH // 2
        draw.ellipse(
            (
                event.x - r, event.y - r,
                event.x + r, event.y + r
            ),
            fill=255
        )

    last_x = event.x
    last_y = event.y


def stop_draw(event):
    global last_x, last_y
    last_x = None
    last_y = None


canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", continue_draw)
canvas.bind("<ButtonRelease-1>", stop_draw)


# =========================
# 6. 清空画布
# =========================
def clear_canvas():
    global image, draw, last_x, last_y

    canvas.delete("all")

    image = Image.new(
        "L",
        (CANVAS_SIZE, CANVAS_SIZE),
        0
    )
    draw = ImageDraw.Draw(image)

    last_x = None
    last_y = None

    result_label.config(text="预测结果：等待书写")
    confidence_label.config(text="")


# =========================
# 7. 预处理并识别数字
# =========================

def recognize():
    # 1. 检查是否写了数字
    if image.getbbox() is None:
        messagebox.showinfo("提示", "请先写一个数字！")
        return

    # 2. 找到笔迹的边界，裁掉周围的大块黑色区域
    bbox = image.getbbox()
    digit = image.crop(bbox)

    # 3. 保持长宽比例，将数字缩放到最大边长 20 像素
    width, height = digit.size
    scale = 20.0 / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    digit = digit.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # 4. 把数字放进 28×28 的黑色图片中央
    # 这比直接把整张画布压缩成 28×28 更接近 MNIST
    processed = Image.new("L", (28, 28), 0)

    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    processed.paste(digit, (x_offset, y_offset))

    # 5. 转成 PyTorch 张量，形状为 [1, 1, 28, 28]
    input_tensor = torch.tensor(
        list(processed.getdata()),
        dtype=torch.float32
    ).reshape(1, 1, 28, 28) / 255.0

    input_tensor = input_tensor.to(device)

    # 6. 使用训练好的 CNN 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

        predicted_digit = outputs.argmax(dim=1).item()
        confidence = probabilities[predicted_digit].item() * 100

    # 7. 显示预测结果
    result_label.config(
        text=f"预测结果：{predicted_digit}",
        fg="#16A34A"
    )

    confidence_label.config(
        text=f"模型置信度：{confidence:.2f}%"
    )

    # 8. 显示概率最高的三个数字，便于观察模型判断
    top_values, top_indices = torch.topk(probabilities, 3)

    details = "   ".join(
        f"{idx.item()}：{value.item() * 100:.1f}%"
        for value, idx in zip(top_values, top_indices)
    )

    probability_label.config(
        text=f"最可能的三个数字：\n{details}"
    )



# =========================
# 8. 界面按钮和结果显示
# =========================
button_frame = tk.Frame(root)
button_frame.pack(pady=8)

recognize_button = tk.Button(
    button_frame,
    text="识别数字",
    font=("Microsoft YaHei", 12),
    width=12,
    command=recognize
)
recognize_button.pack(side=tk.LEFT, padx=8)

clear_button = tk.Button(
    button_frame,
    text="清空画布",
    font=("Microsoft YaHei", 12),
    width=12,
    command=clear_canvas
)
clear_button.pack(side=tk.LEFT, padx=8)

result_label = tk.Label(
    root,
    text="预测结果：等待书写",
    font=("Microsoft YaHei", 18, "bold")
)
result_label.pack(pady=6)

confidence_label = tk.Label(
    root,
    text="",
    font=("Microsoft YaHei", 11)
)
confidence_label.pack(pady=2)

probability_label = tk.Label(
    root,
    text="",
    font=("Consolas", 9),
    wraplength=540,
    justify=tk.CENTER
)
probability_label.pack(pady=5)

footer = tk.Label(
    root,
    text=f"设备：{device} | 模型：MNIST CNN",
    font=("Microsoft YaHei", 9),
    fg="gray"
)
footer.pack(pady=4)

root.mainloop()
