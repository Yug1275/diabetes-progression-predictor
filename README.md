# 🩺 Early Diabetes Progression Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Overview

This project is part of the **AI Lab Course Project (6th Semester)**.
The goal is to predict the **early progression of diabetes** in patients
using Machine Learning techniques, based on medical diagnostic data.

Early detection of diabetes can significantly help in preventive healthcare
by identifying high-risk individuals before the condition worsens.

---

## 🎯 Objectives

- Build a complete ML pipeline to classify patients as diabetic or non-diabetic
- Handle real-world data challenges: missing values, outliers, class imbalance
- Compare 6 ML models and select the best performing one
- Analyze and prevent overfitting using learning curves and regularization
- Provide a user-friendly Streamlit web app for real-time predictions

---

## 📂 Project Structure

```
diabetes-progression-predictor/
├── data/
│   ├── raw/
│   │   └── diabetes.csv              # Original PIMA dataset
│   └── processed/
│       ├── diabetes_cleaned.csv      # Cleaned dataset
│       ├── X_train.npy               # Scaled training features
│       ├── X_test.npy                # Scaled test features
│       ├── y_train.npy               # Training labels (SMOTE balanced)
│       └── y_test.npy                # Test labels
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Phase 2: Dataset exploration
│   ├── 02_data_visualization.ipynb   # Phase 3: EDA & visualizations
│   ├── 03_preprocessing.ipynb        # Phase 4: Data preprocessing
│   ├── 04_model_training.ipynb       # Phase 5: Model training
│   ├── 05_model_evaluation.ipynb     # Phase 6: Model evaluation
│   └── 06_overfitting_analysis.ipynb # Phase 7: Overfitting analysis
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Reusable preprocessing functions
│   ├── model.py                      # Model training utilities
│   ├── evaluate.py                   # Evaluation metric utilities
│   └── visualize.py                  # Visualization utilities
├── models/
│   ├── scaler.pkl                    # Fitted StandardScaler
│   ├── feature_names.pkl             # Feature column names
│   ├── best_model.pkl                # Best model from Phase 6
│   ├── final_best_model.pkl          # Final tuned model from Phase 7
│   └── *.pkl                         # All trained model files
├── reports/
│   ├── 01_class_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_boxplots_outliers.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_violin_plots.png
│   ├── 06_pairplot.png
│   ├── 07_before_after_imputation.png
│   ├── 08_smote_balance.png
│   ├── 09_cv_comparison.png
│   ├── 10_metrics_comparison.png
│   ├── 11_confusion_matrices.png
│   ├── 12_roc_curves.png
│   ├── 13_feature_importance.png
│   ├── 14_evaluation_dashboard.png
│   ├── 15_learning_curves.png
│   ├── 16_validation_curve.png
│   ├── 17_overfitting_fixes.png
│   └── model_comparison.csv
├── app.py                            # Streamlit web application
├── predict.py                        # CLI prediction script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | PIMA Indians Diabetes Dataset |
| **Source** | UCI ML Repository / Kaggle |
| **Samples** | 768 patients |
| **Features** | 8 medical features + 4 engineered = 12 total |
| **Target** | Binary — `1` (Diabetic) / `0` (Non-Diabetic) |
| **Class Split** | 65% Non-Diabetic / 35% Diabetic |

### Feature Description

| Feature | Medical Meaning |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Genetic likelihood based on family history |
| Age | Age in years |
| BMI_Category | Engineered: 0=Underweight, 1=Normal, 2=Overweight, 3=Obese |
| Age_Group | Engineered: 0=Young, 1=Middle-aged, 2=Senior |
| Glucose_Level | Engineered: 0=Normal, 1=Prediabetes, 2=Diabetes range |
| Insulin_Resistance | Engineered: BMI × Glucose / 1000 |

---

## 🤖 ML Models Compared

| Model | Type | CV Accuracy |
|---|---|---|
| Logistic Regression | Baseline linear | ~77% |
| Decision Tree | Rule-based | ~74% |
| Random Forest | Ensemble | ~82% |
| XGBoost | Gradient Boosting | ~83% |
| Support Vector Machine | Kernel-based | ~80% |
| K-Nearest Neighbors | Distance-based | ~76% |

---

## 📈 Results

| Metric | Best Model Score |
|---|---|
| Accuracy | ~85% |
| Precision | ~83% |
| Recall | ~83% |
| F1-Score | ~84% |
| ROC-AUC | ~90% |

> **Key Finding:** Glucose is the most important feature,
> followed by BMI and Age — consistent with medical literature.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| ML Models | scikit-learn, xgboost |
| Class Balancing | imbalanced-learn (SMOTE) |
| Visualization | matplotlib, seaborn, plotly |
| Model Saving | joblib |
| Web App | streamlit |
| Notebooks | jupyter, ipykernel |
| IDE | VS Code + GitHub Copilot |
| Version Control | Git & GitHub |

---

## 🚀 How to Run

### 1. Clone the repository
```powershell
git clone https://github.com/Yug1275/diabetes-progression-predictor.git
cd diabetes-progression-predictor
```

### 2. Create and activate virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> ⚠️ If execution policy error:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Streamlit web app
```powershell
streamlit run app.py
```

### 5. OR run the CLI predictor
```powershell
python predict.py
```

---

## 📋 Project Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Project Setup & Environment | ✅ Done |
| 2 | Dataset Selection & Exploration | ✅ Done |
| 3 | Data Visualization & EDA | ✅ Done |
| 4 | Data Preprocessing & Feature Engineering | ✅ Done |
| 5 | Model Development & Training | ✅ Done |
| 6 | Model Evaluation & Comparison | ✅ Done |
| 7 | Overfitting & Underfitting Analysis | ✅ Done |
| 8 | Prediction Interface (CLI + Web App) | ✅ Done |
| 9 | Documentation & Report | ✅ Done |

---

## 👨‍💻 Author

- **Name:** *Yug Patel and Meet Prajapati*
- **Course:** Artificial Intelligence Lab — 6th Semester
- **Institution:** *Pandit Deendayal Energy University*

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

This tool is developed for **educational purposes only** as part of an
academic project. It does **NOT** replace professional medical advice.
Always consult a qualified healthcare professional for medical decisions.
