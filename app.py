from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

app = Flask(__name__)

# === 1️⃣ Load the full XGBClassifier ===
try:
    model = joblib.load("best_xgboost_model (5).pkl")
    print(f"✅ Loaded XGBClassifier: {type(model)}")
except Exception as e:
    model = None
    print(f"⚠️ Could not load XGBClassifier: {e}")

# === 2️⃣ Load the Booster ===
try:
    booster = xgb.Booster()
    booster.load_model("xgb_booster.json")
    print(f"✅ Loaded Booster: {type(booster)}")
except Exception as e:
    booster = None
    print(f"⚠️ Could not load Booster: {e}")

# === 3️⃣ Recreate scaler ===
scaler = MinMaxScaler()
scaler.data_min_ = np.array([18., 0.])
scaler.data_max_ = np.array([90., 1.])
scaler.data_range_ = scaler.data_max_ - scaler.data_min_
scaler.scale_ = 1 / scaler.data_range_
scaler.min_ = -scaler.data_min_ * scaler.scale_

# === 4️⃣ Feature order ===
feature_order = [
    'Age', 'Osteoporosis',
    'Gender_Female', 'Gender_Male',
    'Hormonal Changes_Normal', 'Hormonal Changes_Postmenopausal',
    'Family History_No', 'Family History_Yes',
    'Smoking_No', 'Smoking_Yes',
    'Alcohol Consumption_Moderate',
    'Medical Conditions_Hyperthyroidism', 'Medical Conditions_Rheumatoid Arthritis',
    'Medications_Corticosteroids',
    'Prior Fractures_No', 'Prior Fractures_Yes'
]

numerical_features = ['Age', 'Osteoporosis']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- 5️⃣ Get user input ---
        if request.form:
            user_input = {}
            for feat in feature_order:
                val = request.form.get(feat)
                if feat in numerical_features:
                    user_input[feat] = float(val) if val else 0.0
                else:
                    user_input[feat] = int(val) if val else 0
        else:
            data = request.get_json(force=True)
            user_input = {feat: data.get(feat, 0) for feat in feature_order}

        # --- 6️⃣ Arrange input ---
        input_vector = np.array([user_input[feat] for feat in feature_order]).reshape(1, -1)

        # Scale numerical parts
        input_vector_scaled = input_vector.copy()
        num_idx = [feature_order.index(f) for f in numerical_features]
        input_vector_scaled[:, num_idx] = scaler.transform(input_vector[:, num_idx])

        # --- 7️⃣ Use XGBClassifier if available ---
        if model:
            prediction = model.predict(input_vector_scaled)
            proba = model.predict_proba(input_vector_scaled).tolist()
            return jsonify({
                'mode': 'XGBClassifier',
                'prediction': int(prediction[0]),
                'probabilities': proba
            })

        # --- 8️⃣ Otherwise, use Booster ---
        elif booster:
            dmatrix = xgb.DMatrix(input_vector_scaled)
            proba = booster.predict(dmatrix)
            prediction = int(proba[0] >= 0.5)  # binary: logistic output
            return jsonify({
                'mode': 'Booster',
                'prediction': prediction,
                'probability': float(proba[0])
            })

        else:
            return jsonify({'error': 'No model or booster loaded!'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
