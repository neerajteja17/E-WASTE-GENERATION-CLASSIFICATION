# E-WASTE-GENERATION-CLASSIFICATION
# ♻️ E-Waste Image Classification with TensorFlow & Gradio

This project classifies images of electronic waste (e.g., batteries, monitors) using a deep learning model (EfficientNetV2B0) built with TensorFlow and deployed using Gradio.

---

## ✅ Features
- Trained on custom image dataset (train/val/test folders)
- Uses EfficientNetV2B0 for transfer learning
- Simple Gradio web interface to test image predictions
- Runs on Apple M1 (MacBook Air) using TensorFlow-Metal

---

## 📦 Requirements

- macOS with Apple Silicon (M1/M2)
- [Miniforge](https://github.com/conda-forge/miniforge) (recommended for M1)
- Python 3.10
- Conda environment with:

```bash
pip install tensorflow-macos tensorflow-metal matplotlib seaborn scikit-learn pillow gradio notebook

