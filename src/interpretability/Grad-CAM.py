import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from torchvision import models, transforms
import torch.nn.functional as F

CLASS_NAMES = ["D-Type Crystal", "L-Type Crystal"]  # 示例：天冬氨酸手性对映体


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.feature_maps = output

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)

        # 获取预测信息
        probs = F.softmax(output, dim=1)
        conf, class_idx = torch.max(probs, 1)

        # 反向传播计算梯度
        loss = output[0, class_idx]
        loss.backward()

        # 生成热力图
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx.item(), conf.item()

    def clear_hooks(self):
        for hook in self.hooks: hook.remove()


class IntegratedVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("SCI 晶体识别系统 - 预测与可视化")
        self.root.geometry("1000x700")

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_path = None

        # --- 控制区 ---
        ctrl_frame = tk.Frame(root)
        ctrl_frame.pack(pady=15)

        tk.Button(ctrl_frame, text="1. 加载模型权重", command=self.load_model, width=15).grid(row=0, column=0, padx=10)
        tk.Button(ctrl_frame, text="2. 选择晶体图像", command=self.load_image, width=15).grid(row=0, column=1, padx=10)
        tk.Button(ctrl_frame, text="3. 启动分析", command=self.analyze, width=15, bg="#2ecc71", fg="white").grid(row=0,
                                                                                                                 column=2,
                                                                                                                 padx=10)

        # --- 结果显示区 (文字) ---
        res_text_frame = tk.Frame(root)
        res_text_frame.pack(pady=5)
        self.res_label = tk.Label(res_text_frame, text="等待分析...", font=("Arial", 14, "bold"), fg="#34495e")
        self.res_label.pack()

        # --- 图像显示区 (左右对比) ---
        self.display_frame = tk.Frame(root)
        self.display_frame.pack(expand=True, fill="both", padx=20)

        self.left_label = tk.Label(self.display_frame, text="[ Original SEM ]", compound="top")
        self.left_label.pack(side="left", expand=True)

        self.right_label = tk.Label(self.display_frame, text="[ Grad-CAM Interpretation ]", compound="top")
        self.right_label.pack(side="right", expand=True)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Weights", "*.pth")])
        if path:
            try:
                # 对应你的 EfficientNet-B2
                self.model = models.efficientnet_b2(num_classes=len(CLASS_NAMES))
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.to(self.device).eval()
                messagebox.showinfo("成功", "模型加载成功！")
            except Exception as e:
                messagebox.showerror("错误", f"加载失败: {e}")

    def load_image(self):
        self.img_path = filedialog.askopenfilename()
        if self.img_path:
            img = Image.open(self.img_path).convert("RGB").resize((450, 450))
            self.show_image(img, self.left_label)

    def show_image(self, pil_img, label_widget):
        photo = ImageTk.PhotoImage(pil_img)
        label_widget.config(image=photo)
        label_widget.image = photo

    def analyze(self):
        if not self.model or not self.img_path:
            messagebox.showwarning("提示", "请确保已加载模型和图像")
            return

        # 预处理
        transform = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        raw_img = Image.open(self.img_path).convert("RGB")
        input_tensor = transform(raw_img).unsqueeze(0).to(self.device)

        # 运行 Grad-CAM & 预测
        cam_tool = GradCAM(self.model, self.model.features[8])
        mask, pred_idx, confidence = cam_tool.generate_cam(input_tensor)
        cam_tool.clear_hooks()

        # 更新结果文字
        class_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"ID: {pred_idx}"
        self.res_label.config(
            text=f"Result: {class_name} | Confidence: {confidence * 100:.2f}%",
            fg="#e67e22" if confidence > 0.8 else "#c0392b"
        )

        # 生成融合图
        img_np = np.array(raw_img.resize((450, 450)))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (450, 450))
        overlay = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

        # 显示融合图
        res_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        self.show_image(res_pil, self.right_label)


if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedVisualizer(root)
    root.mainloop()
