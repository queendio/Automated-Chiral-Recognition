import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import threading
from ultralytics import YOLO

_using_custom_model_path = False
_custom_model_func_from_file = None

try:
    from model2 import efficientnet_b2 as _imported_custom_model_func

    _custom_model_func_from_file = _imported_custom_model_func
    _using_custom_model_path = True
except (ImportError, ModuleNotFoundError):
    pass


def get_classification_model(model_path, num_classes, device):
    """加载 EfficientNet 分类模型"""
    try:
        if _using_custom_model_path and _custom_model_func_from_file is not None:
            model = _custom_model_func_from_file(num_classes=num_classes)
        else:
            model = models.efficientnet_b2(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"加载分类模型失败: {e}")


def process_pipeline(yolo_path, cls_model_path, image_path, num_classes, class_names,
                     conf_threshold=0.35, min_area=0, max_area=float('inf')):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print(f"正在加载 YOLO 模型: {yolo_path}")
    yolo_model = YOLO(yolo_path)

    print(f"正在加载 分类 模型: {cls_model_path}")
    cls_model = get_classification_model(cls_model_path, num_classes, device)

    # 2. 预处理设置
    cls_img_size = 260
    transform = transforms.Compose([
        transforms.Resize((cls_img_size, cls_img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 读取图像
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")

    # 使用 cv2.IMREAD_COLOR 显式读取彩色图
    img_cv_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_cv_bgr is None:
        raise ValueError("无法读取图片，请检查路径或格式")

    # 复制一份用于绘制结果
    img_result = img_cv_bgr.copy()
    h_img, w_img, _ = img_result.shape

    line_thickness = max(2, int(max(h_img, w_img) * 0.003))
    font_scale = max(0.6, max(h_img, w_img) * 0.001)
    font_thickness = max(1, int(line_thickness / 1.5))

    # 4. YOLO 推理
    print("正在运行 YOLO 检测...")
    results = yolo_model.predict(img_cv_bgr, conf=conf_threshold, save=False, verbose=False)
    result = results[0]

    raw_detections_count = len(result.boxes)
    valid_detections_count = 0  # 用于记录过滤后的数量

    print(f"YOLO 原始检测到 {raw_detections_count} 个目标")

    if raw_detections_count == 0:
        return Image.fromarray(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)), "未检测到任何晶体目标"

    # 5. 遍历检测框 -> 过滤 -> 裁剪 -> 分类 -> 标注
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # 边界保护
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # ### 新增功能：面积过滤 ###
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h

        if box_area < min_area:
            # print(f"跳过小框: area={box_area} < {min_area}")
            continue
        if box_area > max_area:
            # print(f"跳过大框: area={box_area} > {max_area}")
            continue

        # 只有通过过滤的框才继续处理
        valid_detections_count += 1

        crop_bgr = img_cv_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0: continue

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        # 分类
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = cls_model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        class_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

        # 标注颜色
        color = (0, 255, 0) if pred_idx == 0 else (0, 0, 255)  # 0类绿色，1类红色

        # 画框
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color, line_thickness)

        # 标签背景与文字
        label_text = f"{class_name} {confidence:.2f}"
        (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        text_y = y1 - 10 if y1 - 10 > h_text else y1 + h_text + 10
        cv2.rectangle(img_result, (x1, text_y - h_text - 5), (x1 + w_text, text_y + 5), color, -1)
        cv2.putText(img_result, label_text, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # 转换回 PIL，保留原始高分辨率
    final_pil = Image.fromarray(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))

    msg = f"处理完成: 原始检测 {raw_detections_count}, 过滤后保留 {valid_detections_count}"
    return final_pil, msg


class CrystalAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("晶体检测与分类系统 (高清版 + 尺寸过滤)")
        master.geometry("1200x950")

        # --- 配置参数 ---
        self.NUM_CLASSES = 2
        self.CLASS_NAMES = ["D", "L"]

        # --- 变量 ---
        self.yolo_path = tk.StringVar()
        self.cls_model_path = tk.StringVar()
        self.image_path = tk.StringVar()

        # ### 新增变量：面积阈值 ###
        self.min_area_var = tk.IntVar(value=300)  # 默认最小面积 100 像素
        self.max_area_var = tk.IntVar(value=10000000)  # 默认很大，相当于不限制

        self._full_yolo_path = ""
        self._full_cls_path = ""
        self._full_img_path = ""

        # 关键变量：用于存储原始高分辨率结果
        self.current_result_image = None

        # --- 界面布局 ---
        self._setup_ui()

    def _setup_ui(self):
        # 控制面板
        control_frame = ttk.LabelFrame(self.master, text="设置面板", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        grid_opts = {'sticky': 'ew', 'padx': 5, 'pady': 5}

        # 1. YOLO 模型选择
        ttk.Label(control_frame, text="1. 分割/检测模型 (YOLO .pt):").grid(row=0, column=0, sticky='w')
        ttk.Button(control_frame, text="浏览...", command=self.select_yolo).grid(row=0, column=1, **grid_opts)
        ttk.Label(control_frame, textvariable=self.yolo_path, foreground="blue").grid(row=0, column=2, sticky='w')

        # 2. 分类模型选择
        ttk.Label(control_frame, text="2. 分类模型 (EffNet .pth):").grid(row=1, column=0, sticky='w')
        ttk.Button(control_frame, text="浏览...", command=self.select_cls_model).grid(row=1, column=1, **grid_opts)
        ttk.Label(control_frame, textvariable=self.cls_model_path, foreground="blue").grid(row=1, column=2, sticky='w')

        # 3. 图片选择
        ttk.Label(control_frame, text="3. 待测图片:").grid(row=2, column=0, sticky='w')
        ttk.Button(control_frame, text="浏览...", command=self.select_image).grid(row=2, column=1, **grid_opts)
        ttk.Label(control_frame, textvariable=self.image_path, foreground="blue").grid(row=2, column=2, sticky='w')

        # ### 新增：面积过滤设置 ###
        filter_frame = ttk.Frame(control_frame)
        filter_frame.grid(row=3, column=0, columnspan=3, sticky='w', pady=5)

        ttk.Label(filter_frame, text="过滤设置 (单位:像素):  ").pack(side=tk.LEFT)
        ttk.Label(filter_frame, text="最小面积:").pack(side=tk.LEFT)
        ttk.Entry(filter_frame, textvariable=self.min_area_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="最大面积:").pack(side=tk.LEFT)
        ttk.Entry(filter_frame, textvariable=self.max_area_var, width=10).pack(side=tk.LEFT, padx=5)

        # --- 修正点：去掉了错误的 list 参数 ---
        ttk.Label(filter_frame, text="(提示: 面积=宽x高)").pack(side=tk.LEFT, padx=10)

        # 4. 运行按钮
        self.run_btn = ttk.Button(control_frame, text="开始检测与分类", command=self.run_thread, state="disabled")
        self.run_btn.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)

        # 5. 保存按钮
        self.save_btn = ttk.Button(control_frame, text="保存高清结果图片", command=self.save_result_image,
                                   state="disabled")
        self.save_btn.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)

        control_frame.columnconfigure(2, weight=1)

        # 显示区域
        display_frame = ttk.Frame(self.master)
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 原图 Label
        self.lbl_orig = ttk.Label(display_frame, text="原图预览", relief="groove", anchor="center")
        self.lbl_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        # 结果图 Label
        self.lbl_res = ttk.Label(display_frame, text="预测结果标注", relief="groove", anchor="center")
        self.lbl_res.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)

        # 状态栏
        self.status_var = tk.StringVar(value="请加载模型和图片")
        self.status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def check_ready(self):
        if self._full_yolo_path and self._full_cls_path and self._full_img_path:
            self.run_btn.config(state="normal")

    def select_yolo(self):
        path = filedialog.askopenfilename(title="选择YOLO模型", filetypes=[("YOLO Models", "*.pt")])
        if path:
            self.yolo_path.set(os.path.basename(path))
            self._full_yolo_path = path
            self.check_ready()

    def select_cls_model(self):
        path = filedialog.askopenfilename(title="选择分类模型", filetypes=[("PyTorch Models", "*.pth")])
        if path:
            self.cls_model_path.set(os.path.basename(path))
            self._full_cls_path = path
            self.check_ready()

    def select_image(self):
        path = filedialog.askopenfilename(title="选择图片", filetypes=[("Images", "*.jpg *.png *.bmp *.jpeg")])
        if path:
            self.image_path.set(os.path.basename(path))
            self._full_img_path = path
            self.show_image(path, self.lbl_orig)
            self.check_ready()
            self.current_result_image = None
            self.lbl_res.config(image='', text="预测结果标注")
            self.save_btn.config(state="disabled")

    def save_result_image(self):
        if self.current_result_image is None:
            messagebox.showwarning("提示", "没有可保存的结果图像。")
            return

        default_name = "annotated_result.jpg"
        if self._full_img_path:
            name, ext = os.path.splitext(os.path.basename(self._full_img_path))
            default_name = f"{name}_result{ext}"

        file_path = filedialog.asksaveasfilename(
            title="保存高清结果图片",
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                self.current_result_image.save(file_path, quality=100, subsampling=0)
                messagebox.showinfo("成功", f"高清图片已保存至:\n{file_path}")
            except Exception as e:
                messagebox.showerror("保存失败", f"保存图片时出错:\n{e}")

    def show_image(self, img_source, label_widget):
        try:
            if isinstance(img_source, str):
                pil_img = Image.open(img_source).convert('RGB')
            else:
                pil_img = img_source

            pil_img_display = pil_img.copy()

            w = label_widget.winfo_width()
            h = label_widget.winfo_height()
            if w < 100: w = 500
            if h < 100: h = 500

            pil_img_display.thumbnail((w, h), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img_display)

            label_widget.config(image=tk_img, text="")
            label_widget.image = tk_img
        except Exception as e:
            print(f"图片显示错误: {e}")

    def run_thread(self):
        try:
            min_a = self.min_area_var.get()
            max_a = self.max_area_var.get()
        except tk.TclError:
            messagebox.showerror("输入错误", "面积阈值必须是整数")
            return

        self.run_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.status_var.set("正在处理中，请稍候...")

        # 将参数传递给 run_process
        threading.Thread(target=self.run_process, args=(min_a, max_a), daemon=True).start()

    def run_process(self, min_area, max_area):
        try:
            final_image, msg = process_pipeline(
                yolo_path=self._full_yolo_path,
                cls_model_path=self._full_cls_path,
                image_path=self._full_img_path,
                num_classes=self.NUM_CLASSES,
                class_names=self.CLASS_NAMES,
                min_area=min_area,  # 传入参数
                max_area=max_area  # 传入参数
            )
            self.master.after(0, lambda: self.update_result(final_image, msg))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.master.after(0, lambda: messagebox.showerror("错误", str(e)))
            self.master.after(0, lambda: self.run_btn.config(state="normal"))
            self.master.after(0, lambda: self.status_var.set("发生错误"))

    def update_result(self, final_image, msg):
        self.current_result_image = final_image
        self.show_image(final_image, self.lbl_res)
        self.status_var.set(msg)
        self.run_btn.config(state="normal")
        self.save_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = CrystalAnalysisApp(root)
    root.mainloop()
