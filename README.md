# Pneumonia Detection with PyTorch (ResNet18)

This project trains a transfer-learning classifier to detect **pneumonia** vs **normal** chest X-rays using a ResNet18 backbone in PyTorch.

## Dataset

The project uses the Kaggle dataset [`paultimothymooney/chest-xray-pneumonia`](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Download and extract:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p ./data
unzip ./data/chest-xray-pneumonia.zip -d ./data
```

Expected folder structure:

```text
data/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

### Class distribution across splits

![Class distribution across splits](src/outputs/assets/class_distribution.png)

Notes:
- The training data is class-imbalanced.
- The validation split is very small (`n=16`), so validation metrics can be noisy.

## Data Pipeline

### 1. Dataset loading
- A custom `PneumoniaDataset` class loads images from class folders.
- Labels are encoded as:
  - `0` for `NORMAL`
  - `1` for `PNEUMONIA`
- Images are converted to RGB and transformed with torchvision.

### 2. Augmentation and preprocessing

Training transform:
- `Resize(256, 256)`
- `RandomResizedCrop(224, scale=(0.8, 1.0))`
- `RandomHorizontalFlip()`
- `RandomRotation(10)`
- `ColorJitter(brightness=0.2, contrast=0.2)`
- `ToTensor()`
- ImageNet normalization

Validation/test transform:
- `Resize(224, 224)`
- `ToTensor()`
- ImageNet normalization

![Sample chest X-ray set](src/outputs/assets/sample_set.png)

### 3. Imbalance handling
- `WeightedRandomSampler` is used for the training loader based on inverse class frequency.
- Class-weighted `CrossEntropyLoss` is also used.

### 4. Training setup
- Batch size: `32`
- Learning rate: `1e-4`
- Max epochs: `20`
- Early stopping patience: `5`
- Optimizer: `AdamW` (`weight_decay=1e-4`)
- LR scheduler: `ReduceLROnPlateau` (`mode="max"`, `factor=0.5`, `patience=2`)

## Model Architecture

Backbone:
- `torchvision.models.resnet18` with ImageNet pretrained weights (`IMAGENET1K_V1`)

Fine-tuning strategy:
- Freeze all layers initially.
- Unfreeze `layer3`, `layer4`, and classifier head for training.

Classification head:
- Replace final FC block with:
  - `Dropout(p=0.3)`
  - `Linear(in_features=512, out_features=2)`

## Results

- Accuracy: `0.9279`
- F1 score: `0.9413`
- ROC-AUC: `0.9808`
- PR-AUC: `0.9869`
- Training run stopped early at epoch `6/20` with best recorded validation accuracy `1.0000` (on `16` validation images).

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.88 | 0.93 | 0.91 | 234 |
| PNEUMONIA | 0.96 | 0.93 | 0.94 | 390 |
| Macro avg | 0.92 | 0.93 | 0.92 | 624 |
| Weighted avg | 0.93 | 0.93 | 0.93 | 624 |

![Evaluation metrics and curves](src/outputs/assets/metrics.png)

The notebook also includes:
- normalized confusion matrix
- ROC curve
- precision-recall curve

## Setup & Usage

### 1) Install dependencies

```bash
uv sync
```

### 2) Open and run `src/resnet18.ipynb`

## Artifacts

- Trained weights: `src/models/model.pth`
