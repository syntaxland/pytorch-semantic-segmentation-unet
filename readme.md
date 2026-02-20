```md
# PyTorch U-Net Semantic Segmentation (Carvana)

Simple **U-Net** implementation in **PyTorch** for binary semantic segmentation, using the **Carvana Image Masking Challenge** dataset.

## Project Structure

```

pytorch-semantic-segmentation-unet/
├─ data/
│  ├─ train_images/
│  ├─ train_masks/
│  ├─ val_images/
│  └─ val_masks/
├─ saved_images/                 # prediction outputs saved during training
├─ dataset.py                    # CarvanaDataset (loads images + masks)
├─ model.py                      # U-Net architecture
├─ train.py                      # training loop + evaluation + saving predictions
├─ utils.py                      # loaders, checkpointing, accuracy + dice
└─ UNET_architecture.png

```

## Dataset

This project expects the Carvana dataset layout:

- Images: `.jpg`
- Masks: corresponding `*_mask.gif` files

Download: Carvana Image Masking Challenge (Kaggle)  
https://www.kaggle.com/competitions/carvana-image-masking-challenge/data

### Place files like this

```

data/train_images/xxx.jpg
data/train_masks/xxx_mask.gif

data/val_images/yyy.jpg
data/val_masks/yyy_mask.gif

````

> Tip: Create a small validation split by moving ~10–20% of training images/masks into `val_*`.

## Install

Create and activate a virtualenv (optional):

```bash
python3 -m venv .venv
source .venv/bin/activate
````

Install dependencies:

```bash
pip install torch torchvision albumentations opencv-python pillow tqdm
```

## Train

Run training:

```bash
python train.py
```

By default, training runs for **3 epochs** and will:

* save checkpoints to `my_checkpoint.pth.tar`
* print **pixel accuracy** and **Dice score**
* save predicted masks into `saved_images/`

## Config

Edit these in `train.py` to match your machine/data:

* `TRAIN_IMG_DIR`, `TRAIN_MASK_DIR`, `VAL_IMG_DIR`, `VAL_MASK_DIR`
* `IMAGE_HEIGHT`, `IMAGE_WIDTH`
* `BATCH_SIZE`, `NUM_EPOCHS`
* `LOAD_MODEL = True/False`

## Output

After training, check:

* `saved_images/pred_*.png` → predicted masks
* `saved_images/*.png` → ground-truth masks (saved alongside)

## Notes

* Loss: `BCEWithLogitsLoss()` (binary segmentation)
* Predictions: `sigmoid` + threshold `> 0.5`
* Metric: Dice score computed in `utils.py`

