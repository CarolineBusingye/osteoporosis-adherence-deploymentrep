from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load trained XGBoost model
model = joblib.load("xgboost_model.pkl")
print(f"Loaded model: {type(model)}")

# Recreate MinMaxScaler for numerical features
scaler = MinMaxScaler()
scaler.data_min_ = np.array([18., 0.])
scaler.data_max_ = np.array([90., 1.])
scaler.data_range_ = scaler.data_max_ - scaler.data_min_
scaler.scale_ = 1 / scaler.data_range_
scaler.min_ = -scaler.data_min_ * scaler.scale_

# Features (must match training order!)
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
        if request.form:
            # HTML form case
            user_input = {}
            for feat in feature_order:
                val = request.form.get(feat)
                if feat in numerical_features:
                    user_input[feat] = float(val) if val else 0.0
                else:
                    user_input[feat] = int(val) if val else 0
        else:
            # JSON API case
            data = request.get_json(force=True)
            user_input = {feat: data.get(feat, 0) for feat in feature_order}

        # Create input array in correct order
        input_vector = np.array([user_input[feat] for feat in feature_order]).reshape(1, -1)

        # Scale only numerical features
        input_vector_scaled = input_vector.copy()
        num_idx = [feature_order.index(f) for f in numerical_features]
        input_vector_scaled[:, num_idx] = scaler.transform(input_vector[:, num_idx])

        # Make prediction
        prediction = model.predict(input_vector_scaled)
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
