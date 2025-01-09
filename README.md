# Vision Transformer vs Compact Convolutional Transformer: CIFAR-10 Benchmark

This repository presents a comparative analysis of **Vision Transformer (ViT)** and **Compact Convolutional Transformer (CCT)** using the CIFAR-10 dataset. The project is implemented in PyTorch and optimized for Google Colab environments.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Training and Evaluation](#training-and-evaluation)
  - [Performance Visualization](#performance-visualization)
  - [Parameter Analysis](#parameter-analysis)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Transformer-based architectures like **ViT** have revolutionized computer vision tasks by adapting the powerful Transformer paradigm from natural language processing. However, their high parameter count and dependency on large datasets have led to the development of more compact and efficient alternatives like **CCT**.

This repository benchmarks these models on the CIFAR-10 dataset, highlighting key differences in:

- Model performance (accuracy and loss trends)
- Parameter count
- Training time

---

## Features

- Implementations of **ViT** and **CCT** from scratch in PyTorch.
- Google Colab-friendly training and evaluation setup.
- Visualization of test loss and accuracy trends.
- Comparison of parameter count and training times for both models.

---

## Requirements

The code is compatible with Python 3.7+ and requires the following libraries:

- PyTorch
- torchvision
- tqdm
- matplotlib

Install the dependencies with:

```bash
pip install torch torchvision tqdm matplotlib
```

---

## Setup
Clone the repository and upload the files to your Google Colab environment:

```bash
git clone https://github.com/your-username/vision-transformer-comparison.git
cd vision-transformer-comparison
```
---
Alternatively, upload the .ipynb file to Colab and run all cells.

## Usage
### Training and Evaluation

Run the following code cells to train both ViT and CCT models on the CIFAR-10 dataset:

```bash
vit_results = train_and_evaluate_model(ViT(), "ViT", train_loader, test_loader, num_epochs)
cct_results = train_and_evaluate_model(CCT(), "CCT", train_loader, test_loader, num_epochs)
```

### Performance Visualization
After training, visualize the test loss and accuracy trends by running the plotting section:

```bash
plt.subplot(1, 2, 1)
plt.plot(epochs, vit_results[1], label="ViT Test Loss")
plt.plot(epochs, cct_results[1], label="CCT Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Comparison")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, vit_results[3], label="ViT Test Accuracy")
plt.plot(epochs, cct_results[3], label="CCT Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")
plt.legend()

plt.tight_layout()
plt.show()
```

### Parameter Analysis
Count the parameters for each model with:

```bash
vit_params = count_parameters(ViT())
cct_params = count_parameters(CCT())

print(f"ViT Parameters: {vit_params}")
print(f"CCT Parameters: {cct_params}")
```
---
## Results

| Metric            | ViT           | CCT           |
|--------------------|---------------|---------------|
| **Parameters**     | 4,542,346     | 333,834       |
| **Test Accuracy**  | ~73%          | ~77%          |
| **Training Time**  | ~653 seconds  | ~662 seconds  |

### Observations

- **Accuracy**: CCT outperforms ViT, likely due to its convolutional tokenization, which better captures local information.
- **Efficiency**: CCT achieves competitive results with significantly fewer parameters.
- **Training Time**: Despite its smaller size, CCT has a similar training time to ViT due to convolutional overhead.


---

## Future Improvements
To further improve the results:

1. **Data Augmentation**:
   - Techniques like AutoAugment, MixUp, and CutMix can enhance generalization.

2. **Learning Rate Schedule**:
   - Use cosine decay schedules to optimize convergence.

3. **Extended Training**:
   - Increase epochs with additional regularization for better accuracy.
