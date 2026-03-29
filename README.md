# ViT-PCOS: Vision Transformer for Automated PCOS Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-100%25-brightgreen?style=for-the-badge)](https://github.com/NoorHammodi/ViT-PCOS)

An advanced deep learning framework leveraging **Vision Transformers (ViT-L-16)** to identify Polycystic Ovary Syndrome (PCOS) from ultrasound imaging with state-of-the-art precision.

---

## 📖 Project Overview
Polycystic Ovary Syndrome (PCOS) affects 8–13% of women of reproductive age, yet nearly 70% remains undiagnosed. This project addresses diagnostic accessibility by automating the classification of ultrasound images into "Infected" (SOPK) and "Non-Infected" categories.

By moving beyond traditional CNN architectures, this model utilizes the global context capabilities of Transformers to achieve superior diagnostic reliability, particularly for underserved communities.

### 🔗 Resources
* [**View Full Project Presentation (PDF)**](INF889G-Presentation.pptx.pdf)

---

## 🧠 Model Architecture
The system employs the **ViT-L-16** architecture, pre-trained on **ImageNet-21k**, featuring a custom-engineered classification head to handle medical imaging complexity.

### 1. Transformer Backbone
* **Model:** Vision Transformer (Large)
* **Patch Size:** 16x16 pixels
* **Input Resolution:** 224x224
* **Mechanism:** Multi-head Self-Attention (MSA) for global feature extraction.

### 2. Custom Classification Head (MLP)
To mitigate "bottlenecks" in feature translation and capture complex diagnostic patterns, a 6-layer sequential MLP was implemented:
* **Architecture:** `2048 → 1024 → 512 → 256 → 128 → 64 → 2 (Classes)`
* **Activation:** ReLU
* **Loss Function:** Cross-Entropy Loss

---

## 📊 Dataset & Results
The model was trained and validated on a balanced dataset of ovary ultrasound images.

### Dataset Distribution
| Split | Infected (SOPK) | Non-Infected | Total |
| :--- | :---: | :---: | :---: |
| **Train** | 781 | 1143 | **1924** |
| **Test** | 787 | 1145 | **1932** |

### Comparative Performance
| Model Architecture | Accuracy (%) |
| :--- | :---: |
| **ViT-PCOS (This Project)** | **100.0%** |
| Standard CNN Baseline | 97.0% |

---

## 🛠️ Technical Implementation
<details>
<summary><b>View Custom Model Class (PyTorch)</b></summary>

```python
import torch.nn as nn
from transformers import ViTModel

class CustomViTForImageClassification(nn.Module):
    def __init__(self, config):
        super(CustomViTForImageClassification, self).__init__()
        self.vit = ViTModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_classes)
        )

    def forward(self, x):
        outputs = self.vit(x)
        # Using the [CLS] token for classification
        return self.classifier(outputs.last_hidden_state[:, 0])
