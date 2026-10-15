"""交互式展示 CNN 如何把手写图像判断为数字。"""

import os
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox, ttk


PROJECT_DIR = Path(__file__).resolve().parent


def _import_torch():
    """导入 PyTorch；误用系统 Python 时自动切换到项目虚拟环境。"""
    try:
        import torch as torch_module
        return torch_module
    except ModuleNotFoundError as error:
        venv_python = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python), *sys.argv])
        raise RuntimeError(
            "未找到 PyTorch。请使用项目虚拟环境运行：\n"
            r".\.venv\Scripts\python.exe convolution_visualizer.py"
        ) from error


torch = _import_torch()
nn = torch.nn


GRID_SIZE = 28
DRAW_CELL = 14
DRAW_SIZE = GRID_SIZE * DRAW_CELL
MAP_SIZE = 168


class MNISTModel(nn.Module):
    """与 mnist_cnn.pth 对应的卷积神经网络。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward_with_activations(self, image):
        conv1 = self.features[1](self.features[0](image))
        pool1 = self.features[2](conv1)
        conv2 = self.features[4](self.features[3](pool1))
        pool2 = self.features[5](conv2)
        logits = self.classifier(pool2)
        return logits, {
            "输入 28×28": image,
            "卷积 1：16×28×28": conv1,
            "池化 1：16×14×14": pool1,
            "卷积 2：32×14×14": conv2,
            "池化 2：32×7×7": pool2,
        }

    def forward(self, image):
        return self.forward_with_activations(image)[0]


def load_model():
    model = MNISTModel()
    model_path = PROJECT_DIR / "mnist_cnn.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件：{model_path}")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


class ConvolutionVisualizer:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.pixels = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32)
        self.activation_canvases = {}
        self.last_cell = None
        self.pending_update = None
        root.title("CNN 手写数字判断过程可视化")
        root.minsize(1120, 720)
        self._build_ui()
        self._render_input()
        self.update_prediction()

    def _build_ui(self):
        ttk.Label(
            self.root, text="CNN 是怎样判断一个手写数字的？",
            font=("Microsoft YaHei UI", 17, "bold"),
        ).pack(pady=(12, 4))
        ttk.Label(
            self.root,
            text="绘制数字后，对比原图、卷积特征和十个候选结果；颜色越亮表示响应越强。",
        ).pack(pady=(0, 10))
        body = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)
        self._build_draw_panel(body)
        self._build_process_panel(body)
        self._build_result_panel(body)

    def _build_draw_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="1. 输入：在这里写数字", padding=10)
        frame.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        self.draw_canvas = tk.Canvas(
            frame, width=DRAW_SIZE, height=DRAW_SIZE, bg="black",
            cursor="crosshair", highlightthickness=1, highlightbackground="#888888",
        )
        self.draw_canvas.pack()
        self.draw_canvas.bind("<Button-1>", self.draw)
        self.draw_canvas.bind("<B1-Motion>", self.draw)
        self.draw_canvas.bind("<ButtonRelease-1>", self.finish_stroke)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(10, 4))
        ttk.Button(buttons, text="清空", command=self.clear).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ttk.Button(buttons, text="立即判断", command=self.update_prediction).pack(
            side="left", expand=True, fill="x"
        )
        ttk.Label(frame, text="提示：用鼠标按住拖动，尽量写在画布中央。",
                  foreground="#555555").pack(anchor="w", pady=(6, 0))

    def _build_process_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="2. 特征提取：逐层看变化", padding=10)
        frame.grid(row=0, column=1, padx=8, sticky="nsew")
        frame.columnconfigure((0, 1), weight=1)
        layer_names = [
            "输入 28×28", "卷积 1：16×28×28", "池化 1：16×14×14",
            "卷积 2：32×14×14", "池化 2：32×7×7",
        ]
        for index, name in enumerate(layer_names):
            cell = ttk.Frame(frame)
            cell.grid(row=index // 2, column=index % 2, padx=5, pady=5, sticky="n")
            ttk.Label(cell, text=name, font=("Microsoft YaHei UI", 9, "bold")).pack()
            canvas = tk.Canvas(
                cell, width=MAP_SIZE, height=MAP_SIZE, bg="black",
                highlightthickness=1, highlightbackground="#bbbbbb",
            )
            canvas.pack(pady=(3, 0))
            self.activation_canvases[name] = canvas
        ttk.Label(
            frame,
            text="卷积寻找笔画/边缘 → 池化保留显著特征并缩小尺寸 → 全连接层综合判断",
            wraplength=380, justify="left", foreground="#444444",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_result_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="3. 分类：比较 0～9", padding=10)
        frame.grid(row=0, column=2, padx=(8, 0), sticky="nsew")
        self.answer_label = ttk.Label(
            frame, text="判断结果：—", font=("Microsoft YaHei UI", 24, "bold")
        )
        self.answer_label.pack(anchor="w")
        self.confidence_label = ttk.Label(frame, text="置信度：—")
        self.confidence_label.pack(anchor="w", pady=(0, 8))
        self.probability_canvas = tk.Canvas(
            frame, width=360, height=330, bg="white",
            highlightthickness=1, highlightbackground="#bbbbbb",
        )
        self.probability_canvas.pack(fill="x")
        self.explanation = ttk.Label(
            frame, text="", wraplength=350, justify="left",
            font=("Microsoft YaHei UI", 10),
        )
        self.explanation.pack(anchor="w", fill="x", pady=(12, 0))

    def draw(self, event):
        col = max(0, min(GRID_SIZE - 1, event.x // DRAW_CELL))
        row = max(0, min(GRID_SIZE - 1, event.y // DRAW_CELL))
        if self.last_cell is None:
            self._paint_brush(row, col)
        else:
            old_row, old_col = self.last_cell
            steps = max(abs(row - old_row), abs(col - old_col), 1)
            for step in range(steps + 1):
                ratio = step / steps
                self._paint_brush(
                    round(old_row + (row - old_row) * ratio),
                    round(old_col + (col - old_col) * ratio),
                )
        self.last_cell = (row, col)
        self._render_input()
        self._schedule_prediction()

    def _paint_brush(self, row, col):
        brush = (
            (-2, 0, .25), (-1, -1, .55), (-1, 0, .85), (-1, 1, .55),
            (0, -2, .25), (0, -1, .85), (0, 0, 1.), (0, 1, .85), (0, 2, .25),
            (1, -1, .55), (1, 0, .85), (1, 1, .55), (2, 0, .25),
        )
        for row_offset, col_offset, value in brush:
            target_row, target_col = row + row_offset, col + col_offset
            if 0 <= target_row < GRID_SIZE and 0 <= target_col < GRID_SIZE:
                self.pixels[target_row, target_col] = max(
                    float(self.pixels[target_row, target_col]), value
                )

    def finish_stroke(self, _event=None):
        self.last_cell = None
        self.update_prediction()

    def _schedule_prediction(self):
        if self.pending_update is not None:
            self.root.after_cancel(self.pending_update)
        self.pending_update = self.root.after(80, self.update_prediction)

    def clear(self):
        self.pixels.zero_()
        self.last_cell = None
        self._render_input()
        self.update_prediction()

    def _render_input(self):
        self.draw_canvas.delete("pixels")
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = float(self.pixels[row, col])
                if value <= 0:
                    continue
                gray = int(value * 255)
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                x1, y1 = col * DRAW_CELL, row * DRAW_CELL
                self.draw_canvas.create_rectangle(
                    x1, y1, x1 + DRAW_CELL, y1 + DRAW_CELL,
                    fill=color, outline=color, tags="pixels",
                )

    def update_prediction(self):
        self.pending_update = None
        image = self.pixels.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits, activations = self.model.forward_with_activations(image)
            probabilities = torch.softmax(logits, dim=1)[0]
        for name, activation in activations.items():
            self._draw_activation(self.activation_canvases[name], activation[0])
        if not torch.any(self.pixels > 0):
            self.answer_label.config(text="判断结果：请先绘制")
            self.confidence_label.config(text="最高置信度：—")
            self._draw_probabilities(torch.zeros(10), -1)
            self.explanation.config(
                text="绘制完成后，这里会展示十个数字的概率、前三名候选以及第一名与第二名的差距。"
            )
            return
        predicted = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted])
        top_values, top_indices = torch.topk(probabilities, 3)
        self.answer_label.config(text=f"判断结果：{predicted}")
        self.confidence_label.config(text=f"最高置信度：{confidence:.1%}")
        self._draw_probabilities(probabilities, predicted)
        ranking = "，".join(
            f"{int(digit)}（{float(value):.1%}）"
            for value, digit in zip(top_values, top_indices)
        )
        margin = float(top_values[0] - top_values[1])
        certainty = "较明确" if margin >= .25 else "存在一些犹豫"
        self.explanation.config(text=(
            "判断过程：网络把最后的 32×7×7 个特征展开并综合打分，再用 Softmax "
            "转成总和为 100% 的概率。\n\n"
            f"前三名：{ranking}。第一名比第二名高 {margin:.1%}，所以这次判断{certainty}。"
        ))

    @staticmethod
    def _draw_activation(canvas, activation):
        canvas.delete("all")
        feature = activation[0] if activation.shape[0] == 1 else activation.abs().mean(dim=0)
        feature = feature.detach().cpu()
        minimum, maximum = float(feature.min()), float(feature.max())
        feature = ((feature - minimum) / (maximum - minimum)
                   if maximum > minimum else torch.zeros_like(feature))
        rows, cols = feature.shape
        cell_width, cell_height = MAP_SIZE / cols, MAP_SIZE / rows
        for row in range(rows):
            for col in range(cols):
                value = float(feature[row, col])
                red, green, blue = int(25 + 230 * value), int(30 + 170 * value), int(55 + 40 * (1 - value))
                canvas.create_rectangle(
                    col * cell_width, row * cell_height,
                    (col + 1) * cell_width + 1, (row + 1) * cell_height + 1,
                    fill=f"#{red:02x}{green:02x}{blue:02x}", outline="",
                )

    def _draw_probabilities(self, probabilities, predicted):
        canvas = self.probability_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 360)
        left, right = 30, width - 62
        max_bar_width = right - left
        for digit, probability_tensor in enumerate(probabilities):
            probability = float(probability_tensor)
            y = 12 + digit * 31
            canvas.create_text(15, y + 10, text=str(digit), font=("Arial", 11, "bold"))
            canvas.create_rectangle(left, y, right, y + 20, fill="#eeeeee", outline="")
            color = "#e45756" if digit == predicted else "#4c78a8"
            canvas.create_rectangle(
                left, y, left + max_bar_width * probability, y + 20, fill=color, outline=""
            )
            canvas.create_text(
                right + 6, y + 10, text=f"{probability:5.1%}", anchor="w", font=("Arial", 9)
            )


def main():
    root = tk.Tk()
    try:
        model = load_model()
    except (FileNotFoundError, RuntimeError) as error:
        root.withdraw()
        messagebox.showerror("模型加载失败", str(error))
        root.destroy()
        return
    ConvolutionVisualizer(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
