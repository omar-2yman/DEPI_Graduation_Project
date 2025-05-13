
# 🩺 Healthcare Predictive Analytics - Heart Disease Risk Prediction

This repository contains a machine learning project focused on **predicting heart disease risk** using clinical and lifestyle-related patient data. The primary goal is to build and evaluate predictive models that aid in early detection and prevention of cardiovascular disease, a leading cause of mortality globally.

---

## 📊 Dataset

The dataset used in this project was sourced from [Kaggle: Heart Disease Risk Prediction Dataset](https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset).

It includes features related to:
- Symptoms (e.g., chest pain, palpitations)
- Clinical indicators (e.g., high cholesterol, blood pressure)
- Lifestyle factors (e.g., smoking, obesity, sedentary lifestyle)
- Medical history (e.g., diabetes, family history)

---

## 🧠 Project Workflow

### 1. **Data Preprocessing**
- Handling missing values
- Feature encoding
- Scaling features for better model performance

### 2. **Exploratory Data Analysis**
- Feature distribution
- Target variable visualization
- Correlation heatmaps

### 3. **Model Development**
Implemented machine learning models include:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Each model is evaluated using:
- Accuracy Score
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve & AUC Score

### 4. **Model Tuning**
- Grid Search for hyperparameter optimization
- Regularization techniques
- Feature scaling

### 5. **Feature Importance**
Using tree-based models to identify the most significant predictors of heart disease.

---

## 🎯 Key Findings

- Symptoms such as chest pain, palpitations, and fatigue are strong predictors.
- Lifestyle-related features like smoking, obesity, and stress contribute significantly.
- Random Forest performed best in terms of accuracy and generalization.

---

## 📂 Repository Structure

```
📁 healthcare-predictive-analytics/
├── data/                  # Raw and cleaned dataset
├── notebooks/             # Jupyter notebooks for EDA and modeling
├── models/                # Trained models and evaluation scripts
├── images/                # Visualizations and plots
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## 👨‍💻 Authors

- Abdelrahman Elsayed Mohamed  
- Abdallah Mohamed Abdallah  
- Mazen Mostafa Abo-ElYazeed  
- Mazen Ehab Gamal  
- Omar Ayman Mohamed  
- Seif El-Eslam Mohamed Yahia

---

## 📌 References

- Kaggle Dataset: [Heart Disease Risk Prediction](https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset)
- Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
- ROC & AUC Explanation: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
