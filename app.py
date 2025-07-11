from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# === Load model ===
model = joblib.load("best_xgboost_model.pkl")
print(f"✅ Loaded: {type(model)}")

# === Define feature order exactly as X_train.columns.tolist() ===
feature_order = [
    'Age',
    'Osteoporosis',
    'Race/Ethnicity',
    'Body Weight',
    'Gender_Female', 'Gender_Male',
    'Hormonal Changes_Normal', 'Hormonal Changes_Postmenopausal',
    'Family History_No', 'Family History_Yes',
    'Smoking_No', 'Smoking_Yes',
    'Alcohol Consumption_Moderate',
    'Medical Conditions_Hyperthyroidism', 'Medical Conditions_Rheumatoid Arthritis',
    'Medications_Corticosteroids',
    'Prior Fractures_No', 'Prior Fractures_Yes'
]

numerical_features = ['Age', 'Osteoporosis', 'Body Weight']  # and others if you scale them!

# === Recreate scaler ===
scaler = MinMaxScaler()
scaler.data_min_ = np.array([18., 0., 40.])  # example: match training min for Age, Osteoporosis, Body Weight
scaler.data_max_ = np.array([90., 1., 150.])
scaler.data_range_ = scaler.data_max_ - scaler.data_min_
scaler.scale_ = 1 / scaler.data_range_
scaler.min_ = -scaler.data_min_ * scaler.scale_

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.form:
            # 🟢 For HTML form
            user_input = {}
            for feat in feature_order:
                val = request.form.get(feat)
                if feat in numerical_features:
                    user_input[feat] = float(val) if val else 0.0
                else:
                    user_input[feat] = int(val) if val else 0
        else:
            # 🟢 For JSON API
            data = request.get_json(force=True)
            user_input = {feat: data.get(feat, 0) for feat in feature_order}

        print(f"🔍 INPUT: {user_input}")

        # === Arrange ===
        input_vector = np.array([user_input[feat] for feat in feature_order]).reshape(1, -1)

        # === Scale numerical ===
        idxs = [feature_order.index(f) for f in numerical_features]
        input_vector_scaled = input_vector.copy()
        input_vector_scaled[:, idxs] = scaler.transform(input_vector[:, idxs])

        # === Predict ===
        pred = model.predict(input_vector_scaled)
        proba = model.predict_proba(input_vector_scaled)

        return jsonify({
            'prediction': int(pred[0]),
            'probability': proba[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
