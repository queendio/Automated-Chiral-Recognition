import os
import math
import argparse
import glob
from PIL import Image
import shutil
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

_using_custom_model_path = False
_custom_model_func_from_file = None

try:
    from model import efficientnet_b2 as _imported_custom_model_func

    _custom_model_func_from_file = _imported_custom_model_func
    _using_custom_model_path = True
except ImportError:
    print(
        "Warning: model.py not found or efficientnet_b2 not in it. Falling back to torchvision.models.efficientnet_b2.")
    from torchvision import models

    _using_custom_model_path = False


def get_model_instance(num_classes_arg, load_imagenet_weights_if_tv):
    if _using_custom_model_path and _custom_model_func_from_file is not None:
        print("Creating model instance using custom efficientnet_b2 from model.py.")
        model = _custom_model_func_from_file(num_classes=num_classes_arg)
    else:
        print("Creating model instance using torchvision.models.efficientnet_b2.")
        tv_weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if load_imagenet_weights_if_tv else None
        model = models.efficientnet_b2(weights=tv_weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes_arg)
    return model


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        self.hooks = []
        self.hooks.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def _forward_hook(self, module, input, output):
        self.feature_maps = output

    def __call__(self, x, class_idx=None):
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()

        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().detach().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def show_cam_on_image(img_pil, mask, colormap=cv2.COLORMAP_JET):
    img_cv = np.array(img_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img_rgb


def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion, num_classes):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    all_labels_epoch = []
    all_preds_epoch = []

    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        loss = criterion(outputs, labels_one_hot)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
        all_labels_epoch.extend(labels.cpu().numpy())
        all_preds_epoch.extend(preds.cpu().numpy())

        if step % 10 == 0:
            print(f"Epoch: {epoch} [{step}/{len(data_loader)}]\tLoss: {loss.item():.4f}")

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    epoch_f1 = f1_score(all_labels_epoch, all_preds_epoch, average='weighted', zero_division=0)
    epoch_mcc = matthews_corrcoef(all_labels_epoch, all_preds_epoch)
    return epoch_loss, epoch_acc.item(), epoch_f1, epoch_mcc


def evaluate(model, data_loader, device, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_labels_epoch = []
    all_preds_epoch = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
            loss = criterion(outputs, labels_one_hot)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_labels_epoch.extend(labels.cpu().numpy())
            all_preds_epoch.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    epoch_f1 = f1_score(all_labels_epoch, all_preds_epoch, average='weighted', zero_division=0)
    epoch_mcc = matthews_corrcoef(all_labels_epoch, all_preds_epoch)
    return epoch_loss, epoch_acc.item(), epoch_f1, epoch_mcc, all_labels_epoch, all_preds_epoch


def load_all_images_and_labels(data_root_path, supported_extensions=None):
    if supported_extensions is None:
        supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    all_image_paths = []
    all_labels = []

    if not os.path.isdir(data_root_path):
        raise FileNotFoundError(f"Data root path not found: {data_root_path}")

    class_names = sorted([d for d in os.listdir(data_root_path) if os.path.isdir(os.path.join(data_root_path, d))])
    if not class_names:
        raise ValueError(f"No class subdirectories found in {data_root_path}")

    class_indices = {name: i for i, name in enumerate(class_names)}
    for class_name, label_idx in class_indices.items():
        class_dir = os.path.join(data_root_path, class_name)
        images_in_class = []
        for ext in supported_extensions:
            images_in_class.extend(glob.glob(os.path.join(class_dir, ext)))
        if not images_in_class:
            print(
                f"Warning: No images found for class '{class_name}' in {class_dir} with extensions {supported_extensions}")
        all_image_paths.extend(images_in_class)
        all_labels.extend([label_idx] * len(images_in_class))

    if not all_image_paths:
        raise ValueError(
            f"No images found in any class subdirectories of {data_root_path}. Please check path and extensions.")
    print(f"Loaded {len(all_image_paths)} images from {len(class_indices)} classes: {class_indices}")
    return all_image_paths, all_labels, class_indices


class PerClassAugmentedDataset(Dataset):


    def __init__(self, image_paths, labels, class_indices, target_size_per_class, transform=None):
        self.transform = transform
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.target_size_per_class = target_size_per_class

        self.paths_by_class = {i: [] for i in range(self.num_classes)}
        for path, label in zip(image_paths, labels):
            if label in self.paths_by_class:
                self.paths_by_class[label].append(path)

        for class_idx, paths in self.paths_by_class.items():
            if not paths:
                class_name = [name for name, idx in self.class_indices.items() if idx == class_idx][0]
                print(f"Warning: Class '{class_name}' (index {class_idx}) has no images and will be skipped.")

        self.num_original_images_by_class = {k: len(v) for k, v in self.paths_by_class.items()}

    def __len__(self):
        return self.num_classes * self.target_size_per_class

    def __getitem__(self, idx):
        if self.__len__() == 0:
            raise IndexError("Dataset is empty.")

        class_idx = idx // self.target_size_per_class

        sample_idx_in_class = idx % self.target_size_per_class

        num_original_for_class = self.num_original_images_by_class.get(class_idx, 0)
        if num_original_for_class == 0:
            raise IndexError(f"Attempting to get item from class {class_idx} which has no original images.")

        original_img_list_idx = sample_idx_in_class % num_original_for_class
        img_path = self.paths_by_class[class_idx][original_img_list_idx]
        label = class_idx

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Check dataset integrity.")
            raise
        except Exception as e:
            print(f"Error opening or converting image {img_path}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        return image, label


def plot_curves(epochs_range, train_metrics, val_metrics, metric_name, save_dir="./plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_metrics, label=f'Train {metric_name}')
    plt.plot(epochs_range, val_metrics, label=f'Validation {metric_name}')
    plt.legend(loc='best')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.savefig(os.path.join(save_dir, f"{metric_name.lower().replace(' ', '_')}_curve.png"))
    plt.close()


def plot_confusion_matrix(true_labels, pred_labels, class_names_list, save_dir="./plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(len(class_names_list)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_list, yticklabels=class_names_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Last Epoch Validation)')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for dir_path in ["./weights", "./plots", "./runs"]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    print(args)

    try:
        print("Loading training data...")
        train_image_paths, train_labels, class_indices = load_all_images_and_labels(args.train_path)
        print("\nLoading validation data...")
        val_image_paths, val_labels, _ = load_all_images_and_labels(args.val_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    train_class_indices_set = set(class_indices.values())
    val_labels_set = set(val_labels)
    if not val_labels_set.issubset(train_class_indices_set):
        print(f"Warning: Validation set contains labels not present in the training set.")
        print(f"Training labels indices: {train_class_indices_set}")
        print(f"Validation labels indices: {val_labels_set}")

    args.num_classes = len(class_indices)
    class_names_list = list(class_indices.keys())
    print(f"Number of classes detected: {args.num_classes}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(args.img_size + 32),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = PerClassAugmentedDataset(train_image_paths, train_labels, class_indices,
                                             args.train_aug_size_per_class, train_transform)
    val_dataset = PerClassAugmentedDataset(val_image_paths, val_labels, class_indices, args.val_aug_size_per_class,
                                           val_transform)

    print(f"Augmented Train Dataset size: {len(train_dataset)}")
    print(f"Augmented Val Dataset size: {len(val_dataset)}")

    nw = min([os.cpu_count() if os.cpu_count() is not None else 0, args.batch_size if args.batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    tb_writer = SummaryWriter(log_dir="./runs/experiment1")
    model = get_model_instance(
        num_classes_arg=args.num_classes,
        load_imagenet_weights_if_tv=(args.weights == "")
    ).to(device)

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            if "state_dict" in weights_dict:
                weights_dict = weights_dict["state_dict"]

            final_classifier_layer_name = None
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential) and len(
                    model.classifier) > 1 and isinstance(model.classifier[1], nn.Linear):
                final_classifier_layer_name = 'classifier.1'
            elif hasattr(model, '_fc') and isinstance(model._fc, nn.Linear):
                final_classifier_layer_name = '_fc'
            if final_classifier_layer_name:
                cls_weight_key = f'{final_classifier_layer_name}.weight'
                cls_bias_key = f'{final_classifier_layer_name}.bias'
                loaded_cls_shape = weights_dict.get(cls_weight_key, torch.empty(0)).shape
                if loaded_cls_shape and loaded_cls_shape[0] != args.num_classes:
                    print(
                        f"Num classes mismatch for classifier. Loaded: {loaded_cls_shape[0]}, Model: {args.num_classes}. Not loading classifier weights.")
                    weights_dict = {k: v for k, v in weights_dict.items() if k not in [cls_weight_key, cls_bias_key]}

            load_weights_dict = {k: v for k, v in weights_dict.items() if
                                 k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
            print(f"Loading weights from {args.weights}. Matched {len(load_weights_dict)}/{len(weights_dict)} keys.")
            missing_keys, unexpected_keys = model.load_state_dict(load_weights_dict, strict=False)
            if missing_keys: print(f"Missing keys: {missing_keys}")
            if unexpected_keys: print(f"Unexpected keys: {unexpected_keys}")
        else:
            print(f"Warning: Weights file {args.weights} not found. Model initialized as per get_model_instance.")

    if args.freeze_layers:
        print("Freezing layers...")
        unfreeze_param_names = []
        if hasattr(model, 'classifier') and isinstance(model.classifier, (nn.Linear, nn.Sequential)):
            for name_part, child_module in model.classifier.named_modules():
                if isinstance(child_module, nn.Linear):
                    prefix = "classifier." + name_part if name_part else "classifier"
                    unfreeze_param_names.extend([prefix + ".weight", prefix + ".bias"])
        elif hasattr(model, '_fc') and isinstance(model._fc, nn.Linear):
            unfreeze_param_names.extend(["_fc.weight", "_fc.bias"])

        if not unfreeze_param_names:
            print("Warning: Could not determine classifier layer names for unfreezing.")
        for name, para in model.named_parameters():
            is_classifier_param = any(name == unf_name for unf_name in unfreeze_param_names)
            if not is_classifier_param:
                para.requires_grad_(False)
            else:
                print(f"Training layer: {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    if not pg:
        print(f"Error: No parameters to optimize. Check freeze_layers logic.")
        return

    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-4)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    train_mccs, val_mccs = [], []
    best_val_acc = 0.0
    last_val_labels, last_val_preds = None, None

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1, train_mcc = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                                                     criterion, args.num_classes)
        scheduler.step()
        val_loss, val_acc, val_f1, val_mcc, val_labels, val_preds = evaluate(model, val_loader, device, criterion,
                                                                            args.num_classes)

        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, MCC: {train_mcc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, MCC: {val_mcc:.4f}"
        )

        train_losses.append(train_loss);
        val_losses.append(val_loss)
        train_accs.append(train_acc);
        val_accs.append(val_acc)
        train_f1s.append(train_f1);
        val_f1s.append(val_f1)
        train_mccs.append(train_mcc)
        val_mccs.append(val_mcc)

        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
        tb_writer.add_scalar("F1_score/train", train_f1, epoch)
        tb_writer.add_scalar("MCC_score/train", train_mcc, epoch)
        tb_writer.add_scalar("Loss/validation", val_loss, epoch)
        tb_writer.add_scalar("Accuracy/validation", val_acc, epoch)
        tb_writer.add_scalar("F1_score/validation", val_f1, epoch)
        tb_writer.add_scalar("MCC_score/validation", val_mcc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")
            print(f"New best validation accuracy: {best_val_acc:.4f}. Model saved.")
        if epoch == args.epochs - 1:
            last_val_labels, last_val_preds = val_labels, val_preds

    tb_writer.close()

    epochs_range = range(args.epochs)
    plot_curves(epochs_range, train_losses, val_losses, "Loss", save_dir="./plots")
    plot_curves(epochs_range, train_accs, val_accs, "Accuracy", save_dir="./plots")
    plot_curves(epochs_range, train_f1s, val_f1s, "F1 Score", save_dir="./plots")
    plot_curves(epochs_range, train_mccs, val_mccs, "MCC Score", save_dir="./plots")
    print("Training curves saved to ./plots/")

    if last_val_labels is not None and last_val_preds is not None:
        plot_confusion_matrix(last_val_labels, last_val_preds, class_names_list, save_dir="./plots")
        print("Confusion matrix saved to ./plots/")

    print("\nGenerating Grad-CAM visualizations...")
    best_model_path = "./weights/best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        try:
            target_layer = model.features[-1]
            print(f"Grad-CAM target layer set to: {type(target_layer)}")
        except (AttributeError, IndexError):
            print("Could not find 'model.features[-1]'. Grad-CAM visualization will be skipped.")
            return

        grad_cam = GradCAM(model=model, target_layer=target_layer)
        num_visualizations = min(10, len(val_image_paths))
        vis_indices = np.random.choice(len(val_image_paths), num_visualizations, replace=False)

        save_dir_gradcam = "./plots/gradcam"
        os.makedirs(save_dir_gradcam, exist_ok=True)

        for i in vis_indices:
            img_path = val_image_paths[i]
            true_label_idx = val_labels[i]
            true_label_name = class_names_list[true_label_idx]

            original_img_pil = Image.open(img_path).convert('RGB').resize((args.img_size, args.img_size))
            tensor_input = val_transform(original_img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(tensor_input)
                pred_label_idx = prediction.argmax(dim=1).item()
                pred_label_name = class_names_list[pred_label_idx]

            mask = grad_cam(tensor_input, class_idx=pred_label_idx)
            superimposed_image = show_cam_on_image(original_img_pil, mask)

            img_name = os.path.basename(img_path)
            save_path = os.path.join(save_dir_gradcam,
                                     f"{os.path.splitext(img_name)[0]}_true-{true_label_name}_pred-{pred_label_name}.png")

            plt.figure(figsize=(8, 8))
            plt.imshow(superimposed_image)
            plt.title(f"True: {true_label_name} | Predicted: {pred_label_name}")
            plt.axis('off')
            plt.savefig(save_path)
            plt.close()

        print(f"Grad-CAM visualizations saved to {save_dir_gradcam}")
        grad_cam.remove_hooks()
    else:
        print("Could not find best model. Skipping Grad-CAM.")

    print("\n--- Training Summary ---")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Validation F1 Score: {val_f1s[-1]:.4f}")
    print(f"Final Validation MCC Score: {val_mccs[-1]:.4f}")
    print("\nTraining finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Image Classification with Augmentation")
    parser.add_argument('--train-path', type=str,
                        default="./data/train",
                        help="Path to the root training data directory (containing class subfolders)")
    parser.add_argument('--val-path', type=str,
                        default="./data/val",
                        help="Path to the root validation data directory (containing class subfolders)")
    parser.add_argument('--train-aug-size-per-class', type=int, default=2000,
                        help="Target number of augmented images per class for the training set")
    parser.add_argument('--val-aug-size-per-class', type=int, default=400,
                        help="Target number of augmented images per class for the validation set")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for all random operations")
    parser.add_argument('--img-size', type=int, default=260,
                        help="Input image size for the model")
    parser.add_argument('--weights', type=str, default="./permodel/efficientnetb2.pth",
                        help='Initial weights path. If empty, uses torchvision pretrained if available.')
    parser.add_argument('--freeze-layers', action='store_true', help="Freeze layers except the classifier head")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=0.01,
                        help="Final LR factor for cosine scheduler")
    parser.add_argument('--device', default='cuda:0', help='Device id (e.g. 0 or cpu)')

    try:
        opt = parser.parse_args([])
    except SystemExit:
        opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    main(opt)
