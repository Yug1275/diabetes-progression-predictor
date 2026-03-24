# 🩺 Early Diabetes Progression Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Overview

This project is part of the **AI Lab Course Project (6th Semester)**.  
The goal is to predict the **early progression of diabetes** in patients using Machine Learning techniques, based on medical diagnostic data.

Early detection of diabetes can significantly help in preventive healthcare by identifying high-risk individuals before the condition worsens.

---

## 🎯 Objectives

- Build a machine learning pipeline to classify patients as diabetic or non-diabetic
- Compare multiple ML models and select the best performing one
- Handle real-world data challenges like missing values and class imbalance
- Provide a simple prediction interface for end users

---

## 📂 Project Structure

```
diabetes-progression-predictor/
├── data/
│   ├── raw/                  # Original downloaded dataset
│   └── processed/            # Cleaned and preprocessed dataset
├── notebooks/                # Jupyter notebooks for EDA and experiments
├── src/
│   ├── preprocessing.py      # Data cleaning and feature engineering
│   ├── model.py              # Model training and saving
│   ├── evaluate.py           # Evaluation metrics and plots
│   └── visualize.py          # EDA visualizations
├── models/                   # Saved trained model files
├── reports/                  # Generated plots and figures
├── app.py                    # Streamlit prediction interface (Phase 8)
├── predict.py                # CLI prediction script (Phase 8)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📊 Dataset

- **Name:** PIMA Indians Diabetes Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) / UCI ML Repository
- **Samples:** 768 patients
- **Features:** 8 medical features (Glucose, BMI, Age, Insulin, Blood Pressure, etc.)
- **Target:** Binary — `1` (Diabetic) / `0` (Non-Diabetic)

---

## 🤖 ML Models Used

| Model | Type |
|---|---|
| Logistic Regression | Baseline |
| Decision Tree | Rule-based |
| Random Forest | Ensemble |
| XGBoost | Gradient Boosting |
| Support Vector Machine | Kernel-based |
| K-Nearest Neighbors | Distance-based |

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Libraries:** pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn, plotly, joblib, streamlit
- **IDE:** VS Code with GitHub Copilot
- **Version Control:** Git & GitHub

## 🚀 How to Run

### 1. Clone the repository
```powershell
git clone https://github.com/<your-username>/diabetes-progression-predictor.git
cd diabetes-progression-predictor
```

### 2. Create and activate virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> ⚠️ If you get an error like *"execution of scripts is disabled"*, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then run the activate command again.

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the prediction app
```powershell
streamlit run app.py
```

## 📈 Results *(Updated after Phase 6)*

> Model evaluation results and comparison will be added here after training.

---

## 📝 Report & Presentation

> Final report and presentation slides will be added in Phase 9.

---

## 👨‍💻 Author

- **Name:** *(Your Name)*
- **Course:** Artificial Intelligence Lab — 6th Semester
- **Institution:** *(Your College Name)*

---

## 📄 License

This project is licensed under the MIT License.