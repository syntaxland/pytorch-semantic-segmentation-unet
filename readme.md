```md
# 🧠 PyTorch U-Net Semantic Segmentation (Carvana)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

A clean and minimal **U-Net implementation in PyTorch** for **binary semantic segmentation**, trained on the **Carvana Image Masking Challenge** dataset.

---

## 📌 Features

- ✅ U-Net architecture from scratch
- ✅ Mixed Precision Training (AMP)
- ✅ Dice Score + Pixel Accuracy
- ✅ Albumentations augmentations
- ✅ Automatic checkpoint saving
- ✅ Prediction image export

---

## 🏗 Project Structure

```

pytorch-semantic-segmentation-unet/
│
├── data/
│   ├── train_images/
│   ├── train_masks/
│   ├── val_images/
│   └── val_masks/
│
├── saved_images/          # Saved predictions
├── dataset.py             # Dataset loader
├── model.py               # U-Net architecture
├── train.py               # Training script
├── utils.py               # Helper functions
└── UNET_architecture.png  # Architecture diagram

```

---

## 📊 Dataset

This project uses:

**Carvana Image Masking Challenge (Kaggle)**  
https://www.kaggle.com/competitions/carvana-image-masking-challenge/data

### Expected Folder Format

```

data/train_images/xxx.jpg
data/train_masks/xxx_mask.gif

data/val_images/yyy.jpg
data/val_masks/yyy_mask.gif

````

> 🔹 Masks must follow naming convention:  
> `image_name.jpg → image_name_mask.gif`

---

## ⚙️ Installation

### 1️⃣ Clone repo

```bash
git clone <your-repo-url>
cd pytorch-semantic-segmentation-unet
````

### 2️⃣ Create virtual environment (optional but recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

Run:

```bash
python train.py
```

During training:

* 📈 Accuracy and Dice score are printed
* 💾 Checkpoints saved as `my_checkpoint.pth.tar`
* 🖼 Predictions saved in `saved_images/`

---

## 🔧 Configuration

You can edit hyperparameters inside `train.py`:

```python
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 3
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
```

To resume training:

```python
LOAD_MODEL = True
```

---

## 📐 Model Details

* Architecture: **U-Net**
* Loss: `BCEWithLogitsLoss`
* Output: Binary mask
* Activation (in eval): `sigmoid`
* Threshold: `0.5`
* Metric: Dice Score

---

## 📷 Example Output

After training:

```
saved_images/
├── pred_0.png
├── pred_1.png
└── ...
```

---

## 🖥 Device Support

Automatically detects:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

Works on:

* ✅ CPU
* ✅ NVIDIA GPU (CUDA)

---

## 🧠 Architecture

See included diagram:

`UNET_architecture.png`

---

## 📜 License

MIT License — free to use and modify.

---

## 🙌 Credits

Based on U-Net implementation inspired by
Aladdin Persson's Machine Learning Collection.

```

---

