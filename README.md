# ChestXray14 1024×1024 VAE + Latent Classifier (No-Skip)

This project trains a convolutional Variational Autoencoder (VAE) on full‑resolution (1024×1024) NIH ChestXray14 images, then uses the learned latent space as input to a simple MLP for multi‑label disease classification.

Main entry point: **`noskip64.ipynb`**

---

## 1. What this actually does

End‑to‑end pipeline:

1. **Load NIH ChestXray14 metadata and images**
   - Reads `Data_Entry_2017_v2020.csv`.
   - Builds absolute image paths and filters out missing files.
   - Creates a binary label `any_abnormal` (0 = `No Finding`, 1 = any pathology).

2. **Multi‑label encoding**
   - Collects all pathology labels (excluding `No Finding`).
   - Builds a `num_classes`‑dim multi‑hot vector for each image based on `"Finding_Labels"`.

3. **Train/val/test split**
   - Stratified splits on `any_abnormal`:
     - ~70% train
     - ~15% validation
     - ~15% test

4. **1024×1024 VAE (no skip connections)**
   - Encoder: 6 downsampling Conv2d blocks from `1×1024×1024` down to `256×16×16`.
   - Latent: linear layers to `mu` and `logvar` with dimension `LATENT_DIM = 64`.
   - Decoder: mirrors encoder using ConvTranspose2d back to `1×1024×1024`.
   - Loss = L1 reconstruction + β * KL (`BETA_KL = 1e‑3`).

5. **Latent extraction**
   - For all train/val/test images:
     - Pass through encoder.
     - Store `mu` (64‑dim latent vector) as feature.
     - Store multi‑hot labels as targets.

6. **Latent‑space classifier**
   - Simple MLP:
     - 64 → 256 → 256 → `num_classes`
     - ReLU activations.
   - Trained with `BCEWithLogitsLoss` for multi‑label classification.
   - Evaluated with macro F1 (per disease + averages) using scikit‑learn.

7. **Visualization**
   - Shows example reconstructions and absolute error maps.
   - t‑SNE of latent space (normal vs “any abnormal”).

If you don’t have a strong GPU with a lot of VRAM, the 1024×1024 setup will be slow and/or may run out of memory. Batch size is **1** for a reason.

---

## 2. Folder and file expectations

The notebook assumes this structure:

```text
ChestXray14/
├── images/
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ...
└── Data_Entry_2017_v2020.csv
