# =============================================================
# predict.py — Command Line Diabetes Prediction Script
# =============================================================
# Usage: python predict.py
# This script loads the trained model and scaler,
# asks the user for medical inputs, and returns a prediction.
# =============================================================

import numpy as np
import joblib
import os

# -------------------------------------------------------
# Load saved model, scaler and feature names
# -------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
model        = joblib.load(os.path.join(BASE_DIR, 'models', 'final_best_model.pkl'))
scaler       = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_names.pkl'))

# -------------------------------------------------------
# Feature input ranges for validation
# (min, max, description)
# -------------------------------------------------------
FEATURE_INFO = {
    'Pregnancies'             : (0,  20,  'Number of pregnancies'),
    'Glucose'                 : (50, 250, 'Plasma glucose concentration (mg/dL)'),
    'BloodPressure'           : (30, 140, 'Diastolic blood pressure (mm Hg)'),
    'SkinThickness'           : (5,  80,  'Triceps skin fold thickness (mm)'),
    'Insulin'                 : (10, 600, '2-Hour serum insulin (mu U/ml)'),
    'BMI'                     : (10, 70,  'Body Mass Index (kg/m²)'),
    'DiabetesPedigreeFunction': (0.05, 2.5,'Diabetes pedigree function score'),
    'Age'                     : (10, 100, 'Age in years'),
    # Engineered features (auto-calculated — not asked from user)
}

def get_engineered_features(inputs):
    """Calculate the 4 engineered features from Phase 4."""
    pregnancies, glucose, bp, skin, insulin, bmi, dpf, age = inputs

    # BMI Category (0=Underweight, 1=Normal, 2=Overweight, 3=Obese)
    if bmi < 18.5:   bmi_cat = 0
    elif bmi < 25:   bmi_cat = 1
    elif bmi < 30:   bmi_cat = 2
    else:            bmi_cat = 3

    # Age Group (0=Young, 1=Middle, 2=Senior)
    if age < 30:     age_grp = 0
    elif age < 45:   age_grp = 1
    else:            age_grp = 2

    # Glucose Level (0=Normal, 1=Prediabetes, 2=Diabetes range)
    if glucose < 100:   gluc_lvl = 0
    elif glucose < 126: gluc_lvl = 1
    else:               gluc_lvl = 2

    # Insulin Resistance Score
    insulin_res = bmi * glucose / 1000

    return bmi_cat, age_grp, gluc_lvl, insulin_res


def get_user_input():
    """Prompt user for all medical inputs with validation."""
    print("\n" + "=" * 55)
    print("   🩺 Diabetes Risk Prediction — Input Form")
    print("=" * 55)
    print("Please enter the following medical values.\n")

    inputs = []
    base_features = list(FEATURE_INFO.keys())

    for feature in base_features:
        min_val, max_val, description = FEATURE_INFO[feature]

        while True:
            try:
                value = float(input(f"  {description}\n  [{feature}] ({min_val}–{max_val}): "))
                if min_val <= value <= max_val:
                    inputs.append(value)
                    break
                else:
                    print(f"  ⚠️  Please enter a value between {min_val} and {max_val}\n")
            except ValueError:
                print("  ⚠️  Please enter a valid number\n")

    return inputs


def predict(inputs):
    """Run prediction on given inputs."""

    # Calculate engineered features
    eng_features = get_engineered_features(inputs)

    # Combine base + engineered features
    all_features = np.array(inputs + list(eng_features)).reshape(1, -1)

    # Scale using the saved scaler
    scaled = scaler.transform(all_features)

    # Get prediction and probability
    prediction   = model.predict(scaled)[0]
    probability  = model.predict_proba(scaled)[0]

    return prediction, probability


def display_result(inputs, prediction, probability):
    """Display the prediction result clearly."""

    risk_pct    = probability[1] * 100
    no_risk_pct = probability[0] * 100

    print("\n" + "=" * 55)
    print("             🔬 PREDICTION RESULT")
    print("=" * 55)

    if prediction == 1:
        print(f"\n  🔴 HIGH RISK OF DIABETES DETECTED")
        print(f"\n  Confidence  : {risk_pct:.1f}%")
        print(f"  Risk Level  : {'🔴 High' if risk_pct > 75 else '🟡 Moderate'}")
        print(f"\n  ⚕️  Recommendation:")
        print(f"     Please consult a healthcare professional")
        print(f"     for a comprehensive diabetes screening.")
    else:
        print(f"\n  🟢 LOW RISK OF DIABETES")
        print(f"\n  Confidence  : {no_risk_pct:.1f}%")
        print(f"  Risk Level  : 🟢 Low")
        print(f"\n  ⚕️  Recommendation:")
        print(f"     Maintain a healthy lifestyle with regular")
        print(f"     exercise and balanced diet.")

    print(f"\n  Probability Breakdown:")
    print(f"    Non-Diabetic : {no_risk_pct:.1f}%  {'█' * int(no_risk_pct//5)}")
    print(f"    Diabetic     : {risk_pct:.1f}%   {'█' * int(risk_pct//5)}")

    print("\n" + "=" * 55)
    print("  ⚠️  DISCLAIMER: This is an AI prediction tool.")
    print("  It does NOT replace professional medical advice.")
    print("=" * 55)


# -------------------------------------------------------
# Main execution
# -------------------------------------------------------
if __name__ == '__main__':
    print("\n  🩺 Early Diabetes Progression Prediction System")
    print("  AI Lab Project — 6th Semester\n")

    while True:
        inputs     = get_user_input()
        pred, prob = predict(inputs)
        display_result(inputs, pred, prob)

        print("\n  Would you like to predict for another patient?")
        again = input("  Enter 'yes' to continue or any key to exit: ").strip().lower()
        if again != 'yes':
            print("\n  Thank you for using the Diabetes Prediction System! 👋\n")
            break