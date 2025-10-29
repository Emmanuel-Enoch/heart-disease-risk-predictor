from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# ---------- Step 1 : Train and save model (only runs once if no model.pkl) ----------
if not os.path.exists("model.pkl"):
    print("🧠 Training model on preprocessed dataset...")
    data = pd.read_csv("heart1.csv")

    # Assuming 'target' column is the label
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    pickle.dump(model, open("model.pkl", "wb"))
    print("✅ Model trained and saved as model.pkl")
else:
    print("✅ Found existing model.pkl — skipping retraining.")

# ---------- Step 2 : Load trained model ----------
model = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs from form
        age = float(request.form['age'])
        sex = 1 if request.form['Sex'] == 'Male' else 0
        cpt = float(request.form['CPT'])
        rbp = float(request.form['RBP'])
        chol = float(request.form['Cholesterol'])
        fbs = float(request.form['FBS'])
        ecg = float(request.form['ECG'])
        mhr = float(request.form['MHR'])
        eia = 1 if request.form['EIA'] == 'Yes' else 0
        std = float(request.form['STD'])
        sts = float(request.form['STS'])
        mvf = float(request.form['MVF'])
        tt = float(request.form['TT'])

        # Combine features into one array (match training order)
        features = np.array([[age, sex, cpt, rbp, chol, fbs, ecg, mhr,
                              eia, std, sts, mvf, tt]])

        # Predict using the model
        prediction = model.predict(features)[0]

        # Convert numeric result to readable text
        result = "High Risk \n u " if prediction == 1 else "Low Risk"

        return render_template(
            'index.html',
            prediction_text=f"Heart Disease Risk: {result}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == '__main__':
    print("🚀 Starting Flask development server...")
    app.run(debug=True)
