# 🍳 SmartRecipe-CNN-Evaluator

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />
</p>

## 📌 Overview
SmartRecipe-CNN-Evaluator is a comparative Deep Learning framework designed to benchmark five distinct Convolutional Neural Network (CNN) architectures for fine-grained food image classification. 

Developed as part of a Senior Design Project, this repository aims to identify the optimal balance between **Top-1 Accuracy** and **Computational Efficiency** (Inference Speed & Model Size) for deployment in a future LLM-based Smart Recipe Assistant. All models are evaluated on the benchmark **Food-101** dataset using a frozen backbone Transfer Learning approach.

## 🚀 Evaluated Architectures
The project analyzes the architectural trade-offs across five deep learning paradigms:
1. **ResNet-50** (Residual/Skip Connections)
2. **EfficientNet-B1** (Compound Scaling)
3. **MobileNetV3-Large** (Depthwise Separable Convolutions / Edge AI)
4. **DenseNet-121** (Dense Feature Concatenation)
5. **VGG-16** (Plain Deep Convolutional Network)

## 📊 Experimental Results (10-Epoch Frozen Backbone)
All models were validated on a validation set of 25,250 images using an NVIDIA T4 GPU. 

| Architecture | Top-1 Accuracy | Macro F1 | Val Loss | Model Size | Inference Speed | Paradigm/Focus |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ResNet-50** | **67.12%** | **0.6667** | **1.2570** | ~97.8 MB | 5.00 it/s | Accuracy Leader |
| **EfficientNet-B1** | 66.02% | 0.6536 | 1.3186 | 30.1 MB | 5.71 it/s | Optimal Efficiency |
| **MobileNetV3** | 61.81% | 0.6126 | 1.4563 | **21.1 MB** | **5.92 it/s** | Edge / Mobile AI |
| **DenseNet-121** | 60.25% | 0.5972 | 1.5228 | ~30.8 MB | 5.38 it/s | Feature Reuse |
| **VGG-16** | 48.70% | 0.4803 | 2.1043 | 528.0 MB | 4.86 it/s | Classic CNN Baseline |

> **Conclusion:** While **ResNet-50** provides the highest raw representational power, **EfficientNet-B1** emerges as the most balanced candidate for production deployment, achieving comparable accuracy with a 69% reduction in model size.

## 📂 Project Structure
```text
SmartRecipe-CNN-Evaluator/
├── src/
│   ├── models/                # Neural network architecture definitions & loaders
│   ├── data_preprocessing.py  # Food-101 dataset ingestion & augmentation pipeline
│   └── evaluate.py            # CLI-based evaluation and metric generation script
├── notebooks/
│   └── SmartRecipe_Evaluation.ipynb # Colab benchmark charts & decision matrices
├── .gitignore                 # Ignores heavy data/weight files
├── requirements.txt           # Environment dependencies
└── README.md
Note on Storage: To ensure a lightweight and agile Git repository, all .pth model checkpoints and the food-101.tar.gz dataset archive are securely hosted on Google Drive rather than being tracked by version control.

⚙️ Installation & Setup
1. Clone the repository
Bash
git clone [https://github.com/gbusray/SmartRecipe-CNN-Evaluator.git](https://github.com/gbusray/SmartRecipe-CNN-Evaluator.git)
cd SmartRecipe-CNN-Evaluator
2. Install dependencies
(It is highly recommended to use a virtual environment.)
Bash
pip install -r requirements.txt
💻 Usage & CLI Execution
To initiate the evaluation pipeline, provide the target model architecture and the exact path to its corresponding .pth weights file via the command line.
Example Execution:
Bash
python src/evaluate.py \
    --model ResNet50 \
    --weights "path/to/weights/resnet50_food101.pth" \
    --batch_size 64
Supported --model Arguments:
  - ResNet50
  - EfficientNetB1
  - MobileNetV3
  - DenseNet121
  - VGG16
