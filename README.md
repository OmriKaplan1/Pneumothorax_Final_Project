# Pneumothorax Detection and Segmentation GUI
This project provides a complete deep learning pipeline for detecting and segmenting pneumothorax in chest X-ray images using Convolutional Neural Networks (CNNs).  
It includes:

- A **GUI application** for loading and analyzing chest X-rays (with optional SpO₂ input)
- A **classification model** (based on Xception + CBAM)
- A **segmentation model** (based on U-Net with ResNet50 backbone)
- Code for training and evaluating both models

---
## 📋 Workflow Overview

> ⚠️ To run the full system (GUI + models), follow these steps in order:

1. **Download the dataset**  
   - [Pneumothorax Chest X-ray Images and Masks (Kaggle)](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks)  
   - Unzip and place in a known directory

2. **Train the models**  
   - Use `xception_train.py` to train the classification model  
   - Use `segmentation_final.py` to train the segmentation model  
   - Models will be saved as `.keras` or `.h5` files

3. **Update model paths in the GUI code**  
   - Open `SmartDoc2.py`  
   - Set the correct file paths to your trained models

4. **Run the GUI**  
   - Launch the app with `python SmartDoc2.py`  
   - Upload X-ray images and get classification + segmentation results

---

## 📦 Repository Structure
├── SmartDoc2.py # GUI application
├── xception_train.py # Classification model training
├── segmentation_final.py # Segmentation model training
├── README.md

## 🚀 Features

- Classify CXR images as **normal** or **pneumothorax**
- Segment affected regions using U-Net
- Accept optional **SpO₂ input** for clinical context
- Interactive GUI with visualization, Grad-CAM heatmaps, and user feedback
- Modular training scripts with editable hyperparameters

---

## 📁 Dataset

This project uses the [Pneumothorax Chest X-ray Images and Masks dataset](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks).

> **Note**: The dataset is **not included** in the repository due to size.  
Please download it separately from Kaggle and place it in your desired working directory.

---

## 🧠 Training the Models

Both training scripts (`xception_train.py` and `segmentation_final.py`) are configurable and can be modified to experiment with different hyperparameters, loss functions, or optimizers.

Install the required libraries using pip:
pip install -r requirements.txt
