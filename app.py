from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load trained model
model = joblib.load("best_xgboost_model (2).pkl")

# Manually reconstruct the MinMaxScaler for 3 numerical features
scaler = MinMaxScaler()
scaler.data_min_ = np.array([18., 0., 0.])
scaler.data_max_ = np.array([90., 1., 1.])
scaler.data_range_ = scaler.data_max_ - scaler.data_min_
scaler.scale_ = 1 / scaler.data_range_
scaler.min_ = -scaler.data_min_ * scaler.scale_

# Full list of model input features, in exact order
feature_order = [
    'Age', 'Osteoporosis', 'Adherence',
    'Gender_Female', 'Gender_Male',
    'Hormonal Changes_Normal', 'Hormonal Changes_Postmenopausal',
    'Family History_No', 'Family History_Yes',
    'Smoking_No', 'Smoking_Yes',
    'Alcohol Consumption_Moderate',
    'Medical Conditions_Hyperthyroidism', 'Medical Conditions_Rheumatoid Arthritis',
    'Medications_Corticosteroids',
    'Prior Fractures_No', 'Prior Fractures_Yes'
]

# Numerical features to be scaled
numerical_features = ['Age', 'Osteoporosis', 'Adherence']

@app.route('/')
def home():
    return render_template('index.html')  # Optional: for form input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Accept inputs via form or JSON
        if request.form:
            user_input = {k: float(request.form.get(k)) for k in feature_order}
        else:
            user_input = request.get_json(force=True)

        # Arrange inputs in the correct order
        input_vector = np.array([user_input[feat] for feat in feature_order]).reshape(1, -1)

        # Scale only numerical columns
        numerical_indices = [feature_order.index(f) for f in numerical_features]
        input_vector_scaled = input_vector.copy()
        input_vector_scaled[:, numerical_indices] = scaler.transform(input_vector[:, numerical_indices])

        # Predict
        prediction = model.predict(input_vector_scaled)
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
