# 👥 K-Nearest Neighbors (KNN) — Classification Project

A supervised machine learning project that applies the **K-Nearest Neighbors (KNN)** algorithm to classify data points based on feature similarity and distance metrics.

This project demonstrates distance-based classification, feature scaling, model selection, and evaluation.

---

## 📌 Overview

This project builds a KNN classifier using a labeled dataset and evaluates how prediction accuracy changes with different values of **K (number of neighbors)**.

The workflow includes preprocessing, scaling, model training, hyperparameter selection, and performance evaluation.

---

## 🎯 Objective

Predict the class label of observations by:

- Measuring distance to nearest neighbors
- Voting based on closest K samples
- Selecting the optimal K value

---

## 🧠 Learning Type

**Supervised Learning — Classification**

- Uses labeled training data  
- Distance-based prediction  
- Instance-based learning method  

---

## 🧩 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📊 Dataset

The dataset contains numeric feature variables and a target class label.

Typical steps included:

- Feature inspection
- Class distribution check
- Train/test split

(Target column name depends on dataset used.)

---

## 🔬 Project Workflow

### 1️⃣ Data Exploration
- Dataset inspection
- Feature distribution analysis
- Correlation checks

---

### 2️⃣ Preprocessing
- Train/test split
- Feature scaling (critical for KNN)
- Standardization using scaler

---

### 3️⃣ Model Training

KNN classifier used:
`sklearn.neighbors.KNeighborsClassifier`

Model trained with multiple K values.

---

### 4️⃣ Choosing Optimal K

- Error rate calculated across K range
- Error vs K plotted
- Best K selected based on lowest error

---

### 5️⃣ Evaluation

Model evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- Error rate comparison

---

## 📈 Results
```
              precision    recall  f1-score   support

           0       0.73      0.72      0.73       148
           1       0.73      0.74      0.74       152

    accuracy                           0.73       300
   macro avg       0.73      0.73      0.73       300
weighted avg       0.73      0.73      0.73       300
```

---

## ⚙️ Key Concepts Demonstrated

- KNN algorithm
- Distance-based learning
- Feature scaling importance
- Hyperparameter tuning (K selection)
- Bias vs variance tradeoff
- Classification metrics

---

## ▶️ How to Run

### Clone repository

```bash
git clone https://github.com/rohitb281/knn-project.git
cd knn-project
```

### Install dependencies
```
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Launch notebook
```
jupyter notebook
```

### Open:
```
K Nearest Neighbors Project.ipynb
```
- Run all cells.

---

#@ 🚀 Possible Improvements
- Distance metric comparison
- Cross-validation for K selection
- Weighted KNN
- Feature selection
- Pipeline automation

---

## ⚠️ Limitations
- Sensitive to feature scaling
- Slower prediction on large datasets
- Performance depends on K choice

---

## 📄 License
- Open for educational and portfolio use.

---

## 👤 Author
- Rohit Bollapragada
- itHub: https://github.com/rohitb281
