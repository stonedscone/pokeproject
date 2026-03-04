# PokéType — Pokémon Type Classification with CNNs

A deep learning research project that classifies Pokémon into one of **17 primary types** based solely on visual features from official artwork, using PyTorch. Seven CNN architectures were implemented, trained from scratch, and benchmarked against each other.

---

## Overview

Can a neural network learn to identify a Pokémon's type just by looking at it? This project explores that question by training multiple CNN architectures on a curated Pokémon image dataset and comparing their performance on a 17-class classification problem.

**Spoiler:** Simpler models won.

---

## Results

| Model | Test Accuracy |
|---|---|
| **CustomCNN** | **30.58%** Best |
| AlexNet | 28.10% |
| Baseline CNN | 27.27% |
| DenseNet121 | 25.62% |
| ResNet18 | 17.36% |
| VGG16 | 14.05% |
| GoogLeNet | 14.05% |

### Best Model — CustomCNN
| Metric | Score |
|---|---|
| Accuracy | 30.58% |
| Precision | 17.42% |
| Recall | 30.58% |
| F1-Score | 21.45% |
| Mean Sensitivity | 18.24% |
| Mean Specificity | 95.39% |

### Best Classified Types (CustomCNN)
- Grass — 75.00%
- Normal — 68.75%
- Fire — 62.50%
- Water — 58.82%

### Hardest Types (All Models)
- Electric — 0.00% across most models
- Ground — 0.00% across most models
- Bug — struggled across all models

---

## Dataset

- **Source:** [Kaggle Pokémon Image Dataset](https://www.kaggle.com/)
- **Generations:** 1–7
- **Total Images:** 806
- **Classes:** 17 types (Flying removed — insufficient samples)
- **Image Size:** 128×128 RGB

| Split | Count |
|---|---|
| Training | 564 |
| Validation | 121 |
| Test | 121 |

---

## Models

All models were trained **from scratch** — no transfer learning or pretrained weights.

| Model | Architecture |
|---|---|
| Baseline CNN | Simple 3-layer CNN |
| AlexNet | 8 layers, ReLU + MaxPool |
| VGG16 | 16 layers, uniform 3×3 convolutions |
| ResNet18 | 18 layers with residual skip connections |
| GoogLeNet | 22 layers with Inception modules |
| DenseNet121 | 121 layers with dense connections |
| CustomCNN | Custom architecture tuned for this dataset |

---

## Key Findings

1. **Simpler is better on small datasets** — Complex architectures (VGG16, GoogLeNet) severely overfit on only 806 images
2. **Visual color cues help** — Fire, Water, and Grass types have strong color associations and were classified most reliably
3. **Some types are visually ambiguous** — Electric and Ground have inconsistent design patterns across Pokémon, making classification nearly impossible without additional context
4. **High specificity, low sensitivity** — All models were conservative in predictions (>94% specificity) but missed many true positives (~18% sensitivity)

---

## Tech Stack

- **Python** — Core language
- **PyTorch** — Model building and training
- **Jupyter Notebook** — Development environment
- **Google Colab** — GPU acceleration (T4)
- **Matplotlib / Seaborn** — Confusion matrices and ROC curves

---

## Getting Started

### Prerequisites
```bash
pip install torch torchvision matplotlib seaborn scikit-learn pandas
```

### Dataset Setup
1. Download the Pokémon dataset from Kaggle
2. Organize images into folders by type
3. Update the dataset path in the notebook

### Run
Open the Jupyter notebook in Google Colab or locally:
```bash
jupyter notebook poketype.ipynb
```

---

## Future Work
- **Transfer learning** — Fine-tuning ImageNet pretrained models could significantly improve accuracy
- **Multi-label classification** — Properly handling dual-type Pokémon
- **GradCAM visualization** — Understanding what features the model actually learns
- **Cross-generation testing** — Does a model trained on Gen 1-7 generalize to newer generations?
- **Larger dataset** — More images per type would allow deeper architectures to shine
