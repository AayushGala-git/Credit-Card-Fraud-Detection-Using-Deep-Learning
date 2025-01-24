# Credit Card Fraud Detection Using Deep Learning

This repository contains the implementation of a **Credit Card Fraud Detection** system using deep learning techniques. The project focuses on identifying fraudulent transactions in credit card data using machine learning and deep learning models.

---

## Project Overview

Fraud detection is a critical component of modern financial systems. This project leverages deep learning to detect fraudulent transactions in a highly imbalanced dataset. The primary focus is on:

1. Handling imbalanced data using techniques like SMOTE.
2. Using a robust neural network for binary classification.
3. Evaluating performance using metrics suited for imbalanced datasets.

---

## Dataset

The dataset used in this project contains transactions labeled as fraudulent (1) or legitimate (0). Key details include:

- Features: 28 numerical columns transformed using PCA.
- Target: Binary labels (0 for legitimate, 1 for fraud).
- File: `creditcard_2023.csv`

Due to confidentiality, the dataset is anonymized.

---

## Features

1. **Preprocessing**:
   - Handle missing values.
   - Normalize and scale numerical features.
   - Address data imbalance using SMOTE.

2. **Deep Learning Model**:
   - Fully connected neural network with dropout layers for regularization.
   - Binary classification using sigmoid activation.

3. **Evaluation**:
   - Evaluate the model using precision, recall, F1-score, and ROC-AUC.
   - Visualize results with confusion matrices and SHAP values.

---

## Model Architecture

The neural network implemented in this project uses the following architecture:

1. **Input Layer**:  
   Accepts 28 features from the dataset.

2. **Hidden Layers**:  
   - Dense layers with ReLU activation.  
   - Dropout layers to prevent overfitting.

3. **Output Layer**:  
   - Dense layer with sigmoid activation for binary classification.

4. **Optimizer**:  
   Adam optimizer for gradient-based learning.

5. **Loss Function**:  
   Binary cross-entropy. 

## Requirements

The following software and libraries are required:

- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- pandas
- matplotlib
- seaborn
- imbalanced-learn
- SHAP

All dependencies are listed in the `requirements.txt` file.

---

## Installation

Follow these steps to set up the project:

```bash
# 1. Clone the repository
git clone https://github.com/AayushGala-git/Credit-Card-Fraud-Detection-Using-Deep-Learning.git
cd Credit-Card-Fraud-Detection-Using-Deep-Learning

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Enable Git Large File Storage (LFS)
git lfs install
git lfs pull
```


