# 🎗️ Breast Cancer Classification using Machine Learning

A machine learning classification project that predicts whether a breast tumor is **Benign** or **Malignant** using diagnostic features from the Breast Cancer Wisconsin dataset.

The project includes data preprocessing, feature scaling, multiple machine learning models, model comparison, evaluation, model serialization, and a Streamlit web application.

---

## 📌 Project Overview

Breast cancer classification is a binary classification problem where the objective is to classify a tumor as:

- 🟢 **Benign**
- 🔴 **Malignant**

In this project, three machine learning algorithms were trained and compared:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

The best-performing model was selected and integrated into a Streamlit application for interactive predictions.

---

## 🎯 Objectives

- Load and explore the breast cancer dataset
- Perform data cleaning and preprocessing
- Remove unnecessary columns
- Prepare features and target variables
- Split the dataset into training and testing sets
- Apply feature scaling
- Train multiple classification models
- Compare model performance
- Select the best-performing model
- Save the trained model and scaler
- Build an interactive Streamlit application
- Generate predictions for new input data

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**.

### Dataset Size

- Total records: **569**
- Original columns: **33**
- Features used: **30**
- Target variable: `diagnosis`

### Target Classes

| Class | Meaning | Count |
|---|---|---:|
| 0 | Benign | 357 |
| 1 | Malignant | 212 |

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### 1. Missing Value Check

Missing values were checked for every column.

The column:

```text
Unnamed: 32
