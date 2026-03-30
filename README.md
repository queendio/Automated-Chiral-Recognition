# PhyChiral: A Physics-Constrained Explainable AI Framework for Chiral Crystal Recognition

[![Paper](https://img.shields.io/badge/Paper-Nature%20Communications-blue)](https://doi.org/[Insert_DOI_Here])
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the **PhyChiral** framework, a physics-constrained and explainable artificial intelligence approach for the automated identification of chiral crystals (D/L-aspartic acid and tyrosine). 

## 🌟 Overview

Distinguishing chiral enantiomers based on crystal morphology has been a challenge since Pasteur’s 1848 discovery. **PhyChiral** addresses this by integrating crystal growth physics with deep learning to achieve high-precision recognition across Scanning Electron Microscopy (SEM) and Optical Microscopy (OM) modalities.

### Core Innovations
- **Physics-Constrained EfficientNet-B2**: A classification backbone optimized with crystallographic constraints.
- **Morphology-Preserving Augmentation**: Utilizes Latent Diffusion Models (Stable Diffusion) to generate realistic synthetic datasets.
- **Multi-Crystal Detection**: A modified YOLO-based pipeline for real-time detection in complex fields of view (FOVs).
- **Explainable AI (XAI)**: Grad-CAM visualization to map model decisions to specific crystal facets.

---

## 📂 Repository Structure

```text
PhyChiral/
├── data/                 # Dataset descriptions and sample images
├── models/               # Pre-trained weights (.pt / .pth)
├── src/                  # Core source code
│   ├── augmentation/     # Stable Diffusion-based generation
│   ├── classification/   # EfficientNet-B2 training & eval
│   ├── detection/        # YOLO detection & FOV synthesis
│   └── interpretability/ # Grad-CAM implementation
├── scripts/              # Command-line interface for inference
├── environment.yml       # Conda environment file
└── README.md             # Project documentation
